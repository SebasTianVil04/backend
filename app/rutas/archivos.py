from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from ..utilidades import archivos_firmados as firmas
from ..utilidades.base_datos import obtener_bd
from ..utilidades.respuesta_rangos import archivo_con_rangos
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.dataset import VideoDataset

router = APIRouter(prefix="/archivos", tags=["Archivos"])


def _tiene_acceso(usuario: Usuario, permiso) -> bool:
    if permiso is None:
        return True
    return bool(usuario.es_admin or usuario.tiene_permiso(permiso))


def _construir_respuesta_url(raiz: str, ruta: str) -> dict:
    token = firmas.generar_token(raiz, ruta)
    return {
        "url": f"/api/v1/archivos/f/{token}",
        "expira_en_segundos": firmas.expiracion_de(raiz),
    }


@router.get("/firmar")
def firmar_archivo(
    raiz: str = Query(..., max_length=50),
    ruta: str = Query(..., max_length=300),
    usuario: Usuario = Depends(obtener_usuario_actual),
):
    if raiz not in firmas.RAICES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Archivo no encontrado")

    if not _tiene_acceso(usuario, firmas.permiso_requerido(raiz)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permisos suficientes")

    if firmas.resolver_ruta(raiz, ruta) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Archivo no encontrado")

    return _construir_respuesta_url(raiz, ruta)


@router.get("/video-dataset/{video_id}/url")
def url_video_dataset(
    video_id: int,
    usuario: Usuario = Depends(obtener_usuario_actual),
    bd: Session = Depends(obtener_bd),
):
    if not _tiene_acceso(usuario, firmas.permiso_requerido("videos_dataset")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permisos suficientes")

    video = bd.query(VideoDataset).filter(VideoDataset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video no encontrado")

    if video.ruta_video:
        nombre = Path(video.ruta_video).name
        if firmas.resolver_ruta("videos_dataset", nombre) is not None:
            return {**_construir_respuesta_url("videos_dataset", nombre), "externo": False}

    if video.drive_url:
        return {"url": video.drive_url, "externo": True}

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Archivo de video no disponible")


@router.get("/f/{token}", include_in_schema=False)
def servir_archivo_firmado(token: str, request: Request):
    datos = firmas.verificar_token(token)
    if datos is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Enlace inválido o expirado")

    raiz, ruta = datos
    archivo = firmas.resolver_ruta(raiz, ruta)
    if archivo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Archivo no encontrado")

    if firmas.permiso_requerido(raiz) is None:
        cache = f"private, max-age={min(firmas.expiracion_de(raiz), 300)}"
    else:
        cache = "no-store"

    return archivo_con_rangos(
        request,
        archivo,
        cache_control=cache,
        cabeceras_extra={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
        },
    )