from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..utilidades.base_datos import obtener_bd
from ..modelos.clase import Clase, TipoVideo
from ..modelos.leccion import Leccion
from ..modelos.usuario import Usuario
from ..esquemas.clase_schemas import ClaseCrear, ClaseActualizar, ClaseRespuesta
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista
from ..utilidades.seguridad import verificar_admin, obtener_usuario_actual

router = APIRouter(prefix="/clases", tags=["Clases"])

@router.post("/", response_model=RespuestaAPI)
def crear_clase(
    clase: ClaseCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        leccion = db.query(Leccion).filter(Leccion.id == clase.leccion_id).first()
        if not leccion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        clase_existente = db.query(Clase).filter(
            Clase.leccion_id == clase.leccion_id,
            Clase.orden == clase.orden
        ).first()
        
        if clase_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe una clase con el orden {clase.orden} en esta lección"
            )
        
        clase_data = clase.dict(exclude_unset=False, exclude_none=False)
        
        if 'tipo_video' in clase_data and clase_data['tipo_video']:
            try:
                clase_data['tipo_video'] = TipoVideo(clase_data['tipo_video'])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tipo de video inválido. Debe ser: {', '.join([e.value for e in TipoVideo])}"
                )
        
        nueva_clase = Clase(**clase_data)
        
        db.add(nueva_clase)
        db.commit()
        db.refresh(nueva_clase)
        
        return RespuestaAPI(
            exito=True,
            mensaje="Clase creada exitosamente",
            datos=ClaseRespuesta.model_validate(nueva_clase).dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al crear clase: {str(e)}"
        )

@router.get("/leccion/{leccion_id}", response_model=RespuestaLista)
def obtener_clases_por_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
        if not leccion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        if usuario_actual.es_admin:
            clases = db.query(Clase).filter(
                Clase.leccion_id == leccion_id
            ).order_by(Clase.orden).all()
        else:
            clases = db.query(Clase).filter(
                Clase.leccion_id == leccion_id,
                Clase.activa == True
            ).order_by(Clase.orden).all()
        
        clases_respuesta = []
        for clase in clases:
            try:
                clase_respuesta = ClaseRespuesta.model_validate(clase)
                clases_respuesta.append(clase_respuesta.dict())
            except Exception:
                continue
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(clases_respuesta)} clases",
            datos=clases_respuesta,
            total=len(clases_respuesta),
            pagina=1,
            por_pagina=len(clases_respuesta)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al obtener clases: {str(e)}"
        )

@router.get("/{clase_id}", response_model=RespuestaAPI)
def obtener_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        if not usuario_actual.es_admin and not clase.activa:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes acceso a esta clase"
            )
        
        return RespuestaAPI(
            exito=True,
            mensaje="Clase obtenida exitosamente",
            datos=ClaseRespuesta.model_validate(clase).dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al obtener clase: {str(e)}"
        )

@router.put("/{clase_id}", response_model=RespuestaAPI)
def actualizar_clase(
    clase_id: int,
    clase_data: ClaseActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        if clase_data.orden is not None and clase_data.orden != clase.orden:
            orden_duplicado = db.query(Clase).filter(
                Clase.leccion_id == clase.leccion_id,
                Clase.orden == clase_data.orden,
                Clase.id != clase_id
            ).first()
            
            if orden_duplicado:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ya existe una clase con el orden {clase_data.orden} en esta lección"
                )
        
        datos_actualizar = clase_data.dict(exclude_unset=True)
        
        if 'tipo_video' in datos_actualizar and datos_actualizar['tipo_video']:
            try:
                datos_actualizar['tipo_video'] = TipoVideo(datos_actualizar['tipo_video'])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tipo de video inválido. Debe ser: {', '.join([e.value for e in TipoVideo])}"
                )
        
        requiere_practica_nuevo = datos_actualizar.get('requiere_practica')
        sena_nueva = datos_actualizar.get('sena')
        
        if requiere_practica_nuevo is not None:
            if requiere_practica_nuevo:
                sena_final = sena_nueva if sena_nueva is not None else clase.sena
                if not sena_final or (isinstance(sena_final, str) and not sena_final.strip()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="La seña es obligatoria cuando se requiere práctica"
                    )
        elif clase.requiere_practica:
            if sena_nueva is not None:
                if not sena_nueva or (isinstance(sena_nueva, str) and not sena_nueva.strip()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="La seña es obligatoria cuando se requiere práctica"
                    )
        
        for campo, valor in datos_actualizar.items():
            setattr(clase, campo, valor)
        
        db.commit()
        db.refresh(clase)
        
        return RespuestaAPI(
            exito=True,
            mensaje="Clase actualizada exitosamente",
            datos=ClaseRespuesta.model_validate(clase).dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al actualizar clase: {str(e)}"
        )

@router.delete("/{clase_id}", response_model=RespuestaAPI)
def eliminar_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        db.delete(clase)
        db.commit()
        
        return RespuestaAPI(
            exito=True,
            mensaje="Clase eliminada exitosamente",
            datos=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al eliminar clase: {str(e)}"
        )