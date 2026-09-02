from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from typing import Optional
import io
from pathlib import Path

from app.utilidades.configuracion import configuracion

SCOPES = ['https://www.googleapis.com/auth/drive']
TOKEN_PATH = Path("token.json")
CLIENT_SECRET_PATH = Path("client_secret.json")

_drive_service = None


def get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    _drive_service = build('drive', 'v3', credentials=creds)
    return _drive_service


def obtener_o_crear_subcarpeta(nombre_subcarpeta: str, carpeta_padre_id: str = None) -> str:
    service = get_drive_service()
    carpeta_padre_id = carpeta_padre_id or configuracion.google_drive_carpeta_id

    query = (
        f"name = '{nombre_subcarpeta}' and "
        f"'{carpeta_padre_id}' in parents and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    resultados = service.files().list(q=query, fields="files(id, name)").execute()
    archivos = resultados.get('files', [])

    if archivos:
        return archivos[0]['id']

    metadata = {
        'name': nombre_subcarpeta,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [carpeta_padre_id],
    }
    carpeta = service.files().create(body=metadata, fields='id').execute()
    return carpeta['id']


def subir_archivo_a_drive(contenido: bytes, nombre_archivo: str, mime_type: str, subcarpeta: str = None) -> dict:
    service = get_drive_service()

    carpeta_destino_id = configuracion.google_drive_carpeta_id
    if subcarpeta:
        carpeta_destino_id = obtener_o_crear_subcarpeta(subcarpeta, carpeta_destino_id)

    metadata = {'name': nombre_archivo, 'parents': [carpeta_destino_id]}
    media = MediaIoBaseUpload(io.BytesIO(contenido), mimetype=mime_type, resumable=True)

    archivo = service.files().create(
        body=metadata, media_body=media, fields='id, name, webViewLink, webContentLink'
    ).execute()

    return {
        'id': archivo['id'],
        'nombre': archivo['name'],
        'url_ver': archivo.get('webViewLink'),
        'url_descarga': f"https://drive.google.com/uc?export=download&id={archivo['id']}",
    }


def descargar_archivo_de_drive(file_id: str) -> bytes:
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buffer.seek(0)
    return buffer.read()


def eliminar_archivo_de_drive(file_id: str) -> bool:
    service = get_drive_service()
    try:
        service.files().delete(fileId=file_id).execute()
        return True
    except Exception as e:
        print(f"Error al eliminar archivo de Drive ({file_id}): {e}")
        return False

def respaldar_modelo_en_drive(ruta_local: Path, subcarpeta: str = "modelos_entrenados") -> Optional[dict]:
    """
    Sube una copia de un modelo .pth a Drive como respaldo,
    sin afectar el archivo local. Se usa justo después de guardar el modelo.
    """
    if not ruta_local.exists():
        print(f"[Drive] No se encontró el archivo para respaldar: {ruta_local}")
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
        print(f"[Drive] Respaldo subido: {ruta_local.name} -> {resultado['url_ver']}")
        return resultado
    except Exception as e:
        print(f"[Drive] Error al respaldar {ruta_local.name}: {e}")
        return None
def descargar_videos_temporalmente(videos_db: list, carpeta_temporal: str = "temp_entrenamiento") -> dict:
    """
    Descarga temporalmente de Drive los videos necesarios para entrenar.
    Recibe una lista de objetos VideoDataset (deben tener drive_file_id).
    Retorna un dict {video_id: ruta_local_temporal}.
    Los videos que no tengan drive_file_id (aún locales) se omiten aquí,
    ya que su ruta_video local sigue siendo válida.
    """
    carpeta = Path(carpeta_temporal)
    carpeta.mkdir(exist_ok=True)

    rutas_temporales = {}

    for video in videos_db:
        if not video.drive_file_id:
            continue

        try:
            contenido = descargar_archivo_de_drive(video.drive_file_id)
            extension = ".mp4" if video.formato == "mp4" else ".webm"
            ruta_local = carpeta / f"{video.id}_{video.sena}{extension}"

            with open(ruta_local, "wb") as f:
                f.write(contenido)

            rutas_temporales[video.id] = str(ruta_local)
            print(f"[Drive] Descargado temporalmente: {ruta_local.name}")

        except Exception as e:
            print(f"[Drive] Error descargando video {video.id} ({video.drive_file_id}): {e}")

    return rutas_temporales


def limpiar_carpeta_temporal(carpeta_temporal: str = "temp_entrenamiento"):
    """Borra la carpeta temporal de entrenamiento tras terminar."""
    import shutil
    carpeta = Path(carpeta_temporal)
    if carpeta.exists():
        try:
            shutil.rmtree(carpeta)
            print(f"[Drive] Carpeta temporal eliminada: {carpeta}")
        except Exception as e:
            print(f"[Drive] Error eliminando carpeta temporal: {e}")