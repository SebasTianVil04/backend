from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.utilidades.base_datos import obtener_bd
from app.modelos.dataset import CategoriaDataset, VideoDataset
from app.esquemas.respuestas import RespuestaAPI, RespuestaLista


router = APIRouter(prefix="/categorias-dataset", tags=["categorias_dataset"])


@router.get("", response_model=RespuestaAPI)
def obtener_categorias(db: Session = Depends(obtener_bd)):
    try:
        # ✅ MODIFICADO: Filtrar solo categorías activas con categoria_id válido
        categorias = db.query(CategoriaDataset).filter(
            CategoriaDataset.activa == True,
            CategoriaDataset.categoria_id != None
        ).all()
        
        datos = []
        for cat in categorias:
            # ✅ NUEVO: Validar que la relación con categoría principal existe
            if not cat.categoria_rel:
                print(f"⚠️ Advertencia: CategoriaDataset ID {cat.id} '{cat.nombre}' no tiene categoría principal vinculada")
                continue
            
            total_videos = db.query(VideoDataset).filter(
                VideoDataset.categoria_id == cat.id
            ).count()
            
            total_frames = db.query(
                func.sum(VideoDataset.frames_extraidos)
            ).filter(
                VideoDataset.categoria_id == cat.id
            ).scalar() or 0
            
            datos.append({
                "id": cat.id,
                "categoria_id": cat.categoria_id,  
                "nombre": cat.nombre,
                "descripcion": cat.descripcion,
                "total_videos": total_videos,
                "total_frames": int(total_frames),
                "activa": cat.activa,
                "fecha_creacion": cat.fecha_creacion.isoformat() if cat.fecha_creacion else None
            })
        
        print(f"✅ Se retornaron {len(datos)} categorías del dataset (de {len(categorias)} encontradas)")
        
        return RespuestaAPI(
            exito=True,
            mensaje="Categorías obtenidas",
            datos=datos
        )
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
