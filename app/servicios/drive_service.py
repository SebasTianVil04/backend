import io
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from app.utilidades.configuracion import configuracion

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]
TIPO_CARPETA = "application/vnd.google-apps.folder"

_bloqueo_credenciales = threading.Lock()
_bloqueo_carpetas = threading.Lock()
_credenciales = None
_local = threading.local()
_carpetas: Dict[str, str] = {}


class DriveNoDisponible(RuntimeError):
    pass


def _escapar(valor: str) -> str:
    return valor.replace("\\", "\\\\").replace("'", "\\'")


def _crear_credenciales():
    if not configuracion.usar_google_drive:
        raise DriveNoDisponible("Google Drive está deshabilitado (USAR_GOOGLE_DRIVE=false)")

    info = configuracion.google_cuenta_servicio_info
    if info:
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    if not (
        configuracion.google_client_id
        and configuracion.google_client_secret
        and configuracion.google_refresh_token
    ):
        raise DriveNoDisponible("Faltan credenciales de Google Drive en las variables de entorno")

    credenciales = Credentials(
        token=None,
        refresh_token=configuracion.google_refresh_token,
        token_uri=configuracion.google_token_uri,
        client_id=configuracion.google_client_id,
        client_secret=configuracion.google_client_secret,
        scopes=SCOPES,
    )
    credenciales.refresh(Request())
    return credenciales


def _obtener_credenciales():
    global _credenciales
    with _bloqueo_credenciales:
        if _credenciales is None:
            _credenciales = _crear_credenciales()
        return _credenciales


def get_drive_service():
    servicio = getattr(_local, "servicio", None)
    if servicio is None:
        servicio = build("drive", "v3", credentials=_obtener_credenciales(), cache_discovery=False)
        _local.servicio = servicio
    return servicio


def obtener_o_crear_subcarpeta(nombre_subcarpeta: str, carpeta_padre_id: Optional[str] = None) -> str:
    service = get_drive_service()
    padre = carpeta_padre_id or configuracion.google_drive_carpeta_id
    if not padre:
        raise DriveNoDisponible("GOOGLE_DRIVE_CARPETA_ID no está configurado")

    clave = f"{padre}/{nombre_subcarpeta}"

    with _bloqueo_carpetas:
        if clave in _carpetas:
            return _carpetas[clave]

        consulta = (
            f"name = '{_escapar(nombre_subcarpeta)}' and "
            f"'{_escapar(padre)}' in parents and "
            f"mimeType = '{TIPO_CARPETA}' and "
            f"trashed = false"
        )
        resultado = service.files().list(
            q=consulta,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        archivos = resultado.get("files", [])

        if archivos:
            carpeta_id = archivos[0]["id"]
        else:
            carpeta = service.files().create(
                body={"name": nombre_subcarpeta, "mimeType": TIPO_CARPETA, "parents": [padre]},
                fields="id",
                supportsAllDrives=True,
            ).execute()
            carpeta_id = carpeta["id"]

        _carpetas[clave] = carpeta_id
        return carpeta_id


def subir_archivo_a_drive(
    contenido: bytes,
    nombre_archivo: str,
    mime_type: str,
    subcarpeta: Optional[str] = None,
) -> dict:
    service = get_drive_service()

    carpeta_destino_id = configuracion.google_drive_carpeta_id
    if not carpeta_destino_id:
        raise DriveNoDisponible("GOOGLE_DRIVE_CARPETA_ID no está configurado")

    if subcarpeta:
        carpeta_destino_id = obtener_o_crear_subcarpeta(subcarpeta, carpeta_destino_id)

    metadata = {"name": nombre_archivo, "parents": [carpeta_destino_id]}
    media = MediaIoBaseUpload(io.BytesIO(contenido), mimetype=mime_type, resumable=True)

    archivo = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, webViewLink",
        supportsAllDrives=True,
    ).execute()

    return {
        "id": archivo["id"],
        "nombre": archivo["name"],
        "url_ver": archivo.get("webViewLink"),
        "url_descarga": f"https://drive.google.com/uc?export=download&id={archivo['id']}",
    }


def descargar_archivo_de_drive(file_id: str) -> bytes:
    service = get_drive_service()
    solicitud = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    descargador = MediaIoBaseDownload(buffer, solicitud)
    terminado = False
    while not terminado:
        _, terminado = descargador.next_chunk()
    buffer.seek(0)
    return buffer.read()


def eliminar_archivo_de_drive(file_id: str) -> bool:
    try:
        get_drive_service().files().delete(fileId=file_id, supportsAllDrives=True).execute()
        return True
    except Exception as e:
        logger.warning("Error al eliminar archivo de Drive (%s): %s", file_id, e)
        return False


def respaldar_modelo_en_drive(ruta_local: Path, subcarpeta: str = "modelos_entrenados") -> Optional[dict]:
    if not ruta_local.exists():
        logger.warning("[Drive] No se encontró el archivo para respaldar: %s", ruta_local)
        return None

    try:
        with open(ruta_local, "rb") as f:
            contenido = f.read()
        resultado = subir_archivo_a_drive(
            contenido,
            ruta_local.name,
            "application/octet-stream",
            subcarpeta=subcarpeta,
        )
        logger.info("[Drive] Respaldo subido: %s -> %s", ruta_local.name, resultado["url_ver"])
        return resultado
    except Exception as e:
        logger.error("[Drive] Error al respaldar %s: %s", ruta_local.name, e)
        return None


def descargar_videos_temporalmente(videos_db: list, carpeta_temporal: str = "temp_entrenamiento") -> dict:
    carpeta = Path(carpeta_temporal)
    carpeta.mkdir(parents=True, exist_ok=True)

    rutas_temporales = {}

    for video in videos_db:
        if not video.drive_file_id:
            continue

        try:
            contenido = descargar_archivo_de_drive(video.drive_file_id)
            extension = ".mp4" if video.formato == "mp4" else ".webm"
            ruta_local = carpeta / f"{int(video.id)}{extension}"

            with open(ruta_local, "wb") as f:
                f.write(contenido)

            rutas_temporales[video.id] = str(ruta_local)
            logger.info("[Drive] Descargado temporalmente: %s", ruta_local.name)

        except Exception as e:
            logger.error("[Drive] Error descargando video %s (%s): %s", video.id, video.drive_file_id, e)

    return rutas_temporales


def limpiar_carpeta_temporal(carpeta_temporal: str = "temp_entrenamiento") -> None:
    carpeta = Path(carpeta_temporal)
    if carpeta.exists():
        try:
            shutil.rmtree(carpeta)
            logger.info("[Drive] Carpeta temporal eliminada: %s", carpeta)
        except Exception as e:
            logger.error("[Drive] Error eliminando carpeta temporal: %s", e)