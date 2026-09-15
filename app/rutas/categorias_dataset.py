from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.utilidades.base_datos import obtener_bd
from app.modelos.dataset import CategoriaDataset, VideoDataset
from app.esquemas.respuestas import RespuestaAPI


router = APIRouter(prefix="/categorias-dataset", tags=["categorias_dataset"])


@router.get("", response_model=RespuestaAPI)
def obtener_categorias(db: Session = Depends(obtener_bd)):
    try:
        categorias = (
            db.query(
                CategoriaDataset,
                func.count(VideoDataset.id).label("total_videos"),
                func.coalesce(func.sum(VideoDataset.frames_extraidos), 0).label("total_frames"),
            )
            .outerjoin(VideoDataset, VideoDataset.categoria_id == CategoriaDataset.id)
            .filter(
                CategoriaDataset.activa.is_(True),
                CategoriaDataset.categoria_id.isnot(None),
            )
            .group_by(CategoriaDataset.id)
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
            for cat, total_videos, total_frames in categorias
            if cat.categoria_rel
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