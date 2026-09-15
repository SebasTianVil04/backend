import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual, verificar_admin
from ..modelos.usuario import Usuario
from ..modelos.categoria import Categoria
from ..modelos.sena_categoria import SenaCategoria
from ..esquemas.sena_categoria import SenaCategoriaCrear, SenaCategoriaActualizar
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista

router = APIRouter(prefix="/categorias/{categoria_id}/senas", tags=["Señas de Categoría"])

DIRECTORIO_REFERENCIAS = Path("archivos_subidos") / "senas_referencia"
DIRECTORIO_REFERENCIAS.mkdir(parents=True, exist_ok=True)

EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
EXTENSIONES_VIDEO = {".mp4", ".webm", ".mov"}
EXTENSIONES_PERMITIDAS = EXTENSIONES_IMAGEN | EXTENSIONES_VIDEO

MAX_TAMANO_IMAGEN = 5 * 1024 * 1024
MAX_TAMANO_VIDEO = 25 * 1024 * 1024


def _obtener_categoria_o_404(categoria_id: int, db: Session) -> Categoria:
    categoria = db.query(Categoria).filter(Categoria.id == categoria_id).first()
    if not categoria:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Categoría {categoria_id} no encontrada")
    return categoria


def _sena_a_dict(sena: SenaCategoria) -> dict:
    return {
        "id": sena.id,
        "categoria_id": sena.categoria_id,
        "nombre": sena.nombre,
        "orden": sena.orden,
        "archivo_referencia": sena.archivo_referencia,
        "tipo_referencia": sena.tipo_referencia,
        "activa": sena.activa,
        "fecha_creacion": sena.fecha_creacion.isoformat() if sena.fecha_creacion else None,
        "fecha_actualizacion": sena.fecha_actualizacion.isoformat() if sena.fecha_actualizacion else None
    }


@router.get("", response_model=RespuestaLista)
def listar_senas(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    _obtener_categoria_o_404(categoria_id, db)

    senas = db.query(SenaCategoria).filter(
        SenaCategoria.categoria_id == categoria_id,
        SenaCategoria.activa == True
    ).order_by(SenaCategoria.orden).all()

    datos = [_sena_a_dict(s) for s in senas]

    return RespuestaLista(
        exito=True,
        mensaje=f"Se encontraron {len(datos)} señas",
        datos=datos,
        total=len(datos),
        pagina=1,
        por_pagina=len(datos)
    )


@router.post("", response_model=RespuestaAPI)
def crear_sena(
    categoria_id: int,
    sena: SenaCategoriaCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    _obtener_categoria_o_404(categoria_id, db)

    existente = db.query(SenaCategoria).filter(
        SenaCategoria.categoria_id == categoria_id,
        func.upper(SenaCategoria.nombre) == sena.nombre.upper()
    ).first()

    if existente:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"La seña '{sena.nombre}' ya existe en esta categoría"
        )

    nueva = SenaCategoria(
        categoria_id=categoria_id,
        nombre=sena.nombre,
        orden=sena.orden,
        activa=True
    )
    db.add(nueva)
    db.commit()
    db.refresh(nueva)

    return RespuestaAPI(exito=True, mensaje="Seña creada exitosamente", datos=_sena_a_dict(nueva))


@router.put("/{sena_id}", response_model=RespuestaAPI)
def actualizar_sena(
    categoria_id: int,
    sena_id: int,
    datos_actualizar: SenaCategoriaActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    sena = db.query(SenaCategoria).filter(
        SenaCategoria.id == sena_id, SenaCategoria.categoria_id == categoria_id
    ).first()

    if not sena:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seña no encontrada")

    for campo, valor in datos_actualizar.dict(exclude_unset=True).items():
        setattr(sena, campo, valor)

    db.commit()
    db.refresh(sena)

    return RespuestaAPI(exito=True, mensaje="Seña actualizada exitosamente", datos=_sena_a_dict(sena))


@router.delete("/{sena_id}", response_model=RespuestaAPI)
def eliminar_sena(
    categoria_id: int,
    sena_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    sena = db.query(SenaCategoria).filter(
        SenaCategoria.id == sena_id, SenaCategoria.categoria_id == categoria_id
    ).first()

    if not sena:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seña no encontrada")

    if sena.archivo_referencia:
        ruta_archivo = Path(sena.archivo_referencia)
        if ruta_archivo.exists():
            try:
                ruta_archivo.unlink()
            except Exception:
                pass

    db.delete(sena)
    db.commit()

    return RespuestaAPI(exito=True, mensaje="Seña eliminada exitosamente", datos={"id": sena_id})


@router.post("/{sena_id}/referencia", response_model=RespuestaAPI)
async def subir_archivo_referencia(
    categoria_id: int,
    sena_id: int,
    archivo: UploadFile = File(...),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    sena = db.query(SenaCategoria).filter(
        SenaCategoria.id == sena_id, SenaCategoria.categoria_id == categoria_id
    ).first()

    if not sena:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seña no encontrada")

    extension = Path(archivo.filename).suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extensión no permitida. Usa: {', '.join(sorted(EXTENSIONES_PERMITIDAS))}"
        )

    es_video = extension in EXTENSIONES_VIDEO
    tipo_referencia = "video" if es_video else "imagen"
    limite_tamano = MAX_TAMANO_VIDEO if es_video else MAX_TAMANO_IMAGEN

    contenido = await archivo.read()
    if len(contenido) > limite_tamano:
        limite_mb = limite_tamano // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El archivo no debe superar {limite_mb}MB"
        )

    if sena.archivo_referencia:
        ruta_anterior = Path(sena.archivo_referencia)
        if ruta_anterior.exists():
            try:
                ruta_anterior.unlink()
            except Exception:
                pass

    nombre_archivo = f"sena_{sena_id}_{uuid.uuid4().hex[:8]}{extension}"
    ruta_destino = DIRECTORIO_REFERENCIAS / nombre_archivo

    with open(ruta_destino, "wb") as f:
        f.write(contenido)

    sena.archivo_referencia = str(ruta_destino)
    sena.tipo_referencia = tipo_referencia
    db.commit()
    db.refresh(sena)

    return RespuestaAPI(
        exito=True,
        mensaje=f"{'Video' if es_video else 'Imagen'} de referencia subida exitosamente",
        datos=_sena_a_dict(sena)
    )


@router.delete("/{sena_id}/referencia", response_model=RespuestaAPI)
def eliminar_archivo_referencia(
    categoria_id: int,
    sena_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    sena = db.query(SenaCategoria).filter(
        SenaCategoria.id == sena_id, SenaCategoria.categoria_id == categoria_id
    ).first()

    if not sena:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seña no encontrada")

    if sena.archivo_referencia:
        ruta_archivo = Path(sena.archivo_referencia)
        if ruta_archivo.exists():
            try:
                ruta_archivo.unlink()
            except Exception:
                pass
        sena.archivo_referencia = None
        sena.tipo_referencia = None
        db.commit()
        db.refresh(sena)

    return RespuestaAPI(exito=True, mensaje="Archivo de referencia eliminado exitosamente", datos=_sena_a_dict(sena))