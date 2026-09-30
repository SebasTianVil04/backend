from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.utilidades.base_datos import obtener_bd
from app.dependencias.permisos import requiere_permiso
from app.modelos.categoria import Categoria
from app.modelos.dataset import CategoriaDataset, VideoDataset
from app.modelos.usuario import Usuario
from app.esquemas.respuestas import RespuestaAPI


router = APIRouter(prefix="/categorias-dataset", tags=["categorias_dataset"])


@router.get("", response_model=RespuestaAPI)
def obtener_categorias(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.ver"))
):
    try:

        totales = (
            db.query(
                VideoDataset.categoria_id.label("categoria_id"),
                func.count(VideoDataset.id).label("total_videos"),
                func.coalesce(func.sum(VideoDataset.frames_extraidos), 0).label("total_frames"),
            )
            .group_by(VideoDataset.categoria_id)
            .subquery()
        )

        filas = (
            db.query(
                CategoriaDataset,
                func.coalesce(totales.c.total_videos, 0),
                func.coalesce(totales.c.total_frames, 0),
            )

            .join(Categoria, Categoria.id == CategoriaDataset.categoria_id)
            .outerjoin(totales, totales.c.categoria_id == CategoriaDataset.id)
            .filter(CategoriaDataset.activa.is_(True))
            .order_by(CategoriaDataset.orden, CategoriaDataset.id)
            .all()
        )

        datos = [
            {
                "id": cat.id,
                "categoria_id": cat.categoria_id,
                "nombre": cat.nombre,
                "descripcion": cat.descripcion,
                "total_videos": int(total_videos),
                "total_frames": int(total_frames),
                "activa": cat.activa,
                "fecha_creacion": cat.fecha_creacion.isoformat() if cat.fecha_creacion else None,
            }
            for cat, total_videos, total_frames in filas
        ]

        return RespuestaAPI(
            exito=True,
            mensaje="Categorías obtenidas",
            datos=datos,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )