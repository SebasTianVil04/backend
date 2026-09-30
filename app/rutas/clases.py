import os
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..dependencias.permisos import requiere_permiso
from ..esquemas.clase_schemas import ClaseActualizar, ClaseCrear, ClaseRespuesta
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista
from ..modelos.clase import Clase, TipoVideo
from ..modelos.leccion import Leccion
from ..modelos.usuario import Usuario
from ..utilidades import archivos_firmados as firmas
from ..utilidades.base_datos import obtener_bd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clases", tags=["Clases"])

TIPOS_VIDEO_VALIDOS = ", ".join(tipo.value for tipo in TipoVideo)

RAIZ_VIDEOS = "videos_lecciones"
EXTENSIONES_VIDEO_CLASE = {"mp4", "webm"}
TAMANO_MINIMO_VIDEO = 1024
TAMANO_MAXIMO_VIDEO = 300 * 1024 * 1024
VIGENCIA_VIDEO_SEGUNDOS = 4 * 3600


def _confirmar(db: Session) -> None:
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def _serializar(clase: Clase) -> dict:
    return ClaseRespuesta.model_validate(clase).model_dump()


def _obtener_clase_o_404(db: Session, clase_id: int) -> Clase:
    clase = db.query(Clase).filter(Clase.id == clase_id).first()
    if not clase:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clase no encontrada")
    return clase


def _validar_orden_libre(db: Session, leccion_id: int, orden: int, excluir_id: int | None = None) -> None:
    query = db.query(Clase.id).filter(Clase.leccion_id == leccion_id, Clase.orden == orden)
    if excluir_id is not None:
        query = query.filter(Clase.id != excluir_id)
    if query.first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ya existe una clase con el orden {orden} en esta lección",
        )


def _normalizar_tipo_video(datos: dict) -> None:
    tipo = datos.get("tipo_video")
    if not tipo:
        return
    try:
        datos["tipo_video"] = TipoVideo(tipo)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de video inválido. Debe ser: {TIPOS_VIDEO_VALIDOS}",
        )


def _validar_sena_practica(requiere_practica: bool | None, sena: str | None) -> None:
    if requiere_practica and not (sena or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La seña es obligatoria cuando se requiere práctica",
        )


def _directorio_videos() -> Path:
    directorio = firmas.RAICES[RAIZ_VIDEOS]["ruta"]
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


def _borrar_video_local(nombre: str | None) -> None:
    if not nombre:
        return
    try:
        ruta = _directorio_videos() / Path(nombre).name
        if ruta.is_file():
            ruta.unlink()
    except OSError as e:
        logger.warning("No se pudo eliminar el video local %s: %s", nombre, e)


def _tamano_archivo(archivo: UploadFile) -> int:
    archivo.file.seek(0, os.SEEK_END)
    tamano = archivo.file.tell()
    archivo.file.seek(0)
    return tamano


def _firma_valida(cabecera: bytes, extension: str) -> bool:
    if extension == "mp4":
        return cabecera[4:8] == b"ftyp"
    return cabecera[:4] == b"\x1a\x45\xdf\xa3"


@router.post("/", response_model=RespuestaAPI)
def crear_clase(
    clase: ClaseCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.crear")),
):
    if not db.query(Leccion.id).filter(Leccion.id == clase.leccion_id).first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lección no encontrada")

    _validar_orden_libre(db, clase.leccion_id, clase.orden)

    datos = clase.model_dump()
    _normalizar_tipo_video(datos)
    _validar_sena_practica(datos.get("requiere_practica"), datos.get("sena"))

    nueva_clase = Clase(**datos)
    db.add(nueva_clase)
    _confirmar(db)
    db.refresh(nueva_clase)

    return RespuestaAPI(exito=True, mensaje="Clase creada exitosamente", datos=_serializar(nueva_clase))


@router.get("/leccion/{leccion_id}", response_model=RespuestaLista)
def obtener_clases_por_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.ver")),
):
    if not db.query(Leccion.id).filter(Leccion.id == leccion_id).first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lección no encontrada")

    query = db.query(Clase).filter(Clase.leccion_id == leccion_id)
    if not usuario_actual.es_admin:
        query = query.filter(Clase.activa.is_(True))

    datos = []
    for clase in query.order_by(Clase.orden).all():
        try:
            datos.append(_serializar(clase))
        except ValidationError:
            logger.warning("Clase %s omitida: no cumple el esquema de respuesta", clase.id, exc_info=True)

    return RespuestaLista(
        exito=True,
        mensaje=f"Se encontraron {len(datos)} clases",
        datos=datos,
        total=len(datos),
        pagina=1,
        por_pagina=len(datos),
    )


@router.get("/{clase_id}", response_model=RespuestaAPI)
def obtener_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.ver")),
):
    clase = _obtener_clase_o_404(db, clase_id)

    if not usuario_actual.es_admin and not clase.activa:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes acceso a esta clase")

    return RespuestaAPI(exito=True, mensaje="Clase obtenida exitosamente", datos=_serializar(clase))


@router.get("/{clase_id}/video", response_model=RespuestaAPI)
def obtener_video_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.ver")),
):
    clase = _obtener_clase_o_404(db, clase_id)

    if not usuario_actual.es_admin and not clase.activa:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes acceso a esta clase")

    if clase.video_archivo and firmas.resolver_ruta(RAIZ_VIDEOS, clase.video_archivo) is not None:
        token = firmas.generar_token(RAIZ_VIDEOS, clase.video_archivo, segundos=VIGENCIA_VIDEO_SEGUNDOS)
        return RespuestaAPI(
            exito=True,
            mensaje="Video obtenido exitosamente",
            datos={
                "tipo": "local",
                "url": f"/api/v1/archivos/f/{token}",
                "expira_en_segundos": VIGENCIA_VIDEO_SEGUNDOS,
            },
        )

    url_externa = clase.url_video_embebida
    if url_externa:
        return RespuestaAPI(
            exito=True,
            mensaje="Video obtenido exitosamente",
            datos={"tipo": "externo", "url": url_externa, "expira_en_segundos": None},
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Esta clase no tiene video")


@router.post("/{clase_id}/video", response_model=RespuestaAPI)
def subir_video_clase(
    clase_id: int,
    archivo: UploadFile = File(...),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.editar")),
):
    clase = _obtener_clase_o_404(db, clase_id)

    extension = (archivo.filename or "").rsplit(".", 1)[-1].lower()
    if extension not in EXTENSIONES_VIDEO_CLASE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formato de video no permitido. Usa MP4 o WebM")

    tamano = _tamano_archivo(archivo)
    if tamano < TAMANO_MINIMO_VIDEO:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo de video está vacío o es demasiado pequeño")
    if tamano > TAMANO_MAXIMO_VIDEO:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="El video supera el tamaño máximo permitido")

    cabecera = archivo.file.read(16)
    archivo.file.seek(0)
    if not _firma_valida(cabecera, extension):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El contenido no corresponde a un video válido")

    nombre = f"{uuid.uuid4().hex}.{extension}"
    destino = _directorio_videos() / nombre

    try:
        with open(destino, "wb") as salida:
            shutil.copyfileobj(archivo.file, salida, 1024 * 1024)
    except Exception as e:
        _borrar_video_local(nombre)
        logger.error("Error guardando video de clase %s: %s", clase_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error guardando el video")

    anterior = clase.video_archivo
    clase.video_archivo = nombre

    try:
        _confirmar(db)
    except Exception as e:
        _borrar_video_local(nombre)
        logger.error("Error registrando video de clase %s: %s", clase_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error registrando el video")

    _borrar_video_local(anterior)

    logger.info("Video de clase %s actualizado por usuario %s", clase_id, usuario_actual.id)

    return RespuestaAPI(
        exito=True,
        mensaje="Video de la clase actualizado exitosamente",
        datos={"clase_id": clase_id, "tamano_bytes": tamano},
    )


@router.delete("/{clase_id}/video", response_model=RespuestaAPI)
def eliminar_video_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.editar")),
):
    clase = _obtener_clase_o_404(db, clase_id)

    if not clase.video_archivo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Esta clase no tiene video propio")

    anterior = clase.video_archivo
    clase.video_archivo = None
    _confirmar(db)
    _borrar_video_local(anterior)

    return RespuestaAPI(exito=True, mensaje="Video de la clase eliminado", datos=None)


@router.put("/{clase_id}", response_model=RespuestaAPI)
def actualizar_clase(
    clase_id: int,
    clase_data: ClaseActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.editar")),
):
    clase = _obtener_clase_o_404(db, clase_id)
    datos = clase_data.model_dump(exclude_unset=True)

    orden = datos.get("orden")
    if orden is not None and orden != clase.orden:
        _validar_orden_libre(db, clase.leccion_id, orden, excluir_id=clase_id)

    _normalizar_tipo_video(datos)

    if "requiere_practica" in datos or "sena" in datos:
        _validar_sena_practica(
            datos.get("requiere_practica", clase.requiere_practica),
            datos.get("sena", clase.sena),
        )

    for campo, valor in datos.items():
        setattr(clase, campo, valor)

    _confirmar(db)
    db.refresh(clase)

    return RespuestaAPI(exito=True, mensaje="Clase actualizada exitosamente", datos=_serializar(clase))


@router.delete("/{clase_id}", response_model=RespuestaAPI)
def eliminar_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("clases.eliminar")),
):
    clase = _obtener_clase_o_404(db, clase_id)
    video_local = clase.video_archivo

    db.delete(clase)
    _confirmar(db)
    _borrar_video_local(video_local)

    return RespuestaAPI(exito=True, mensaje="Clase eliminada exitosamente", datos=None)