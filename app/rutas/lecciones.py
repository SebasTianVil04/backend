from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from typing import Optional

from app.modelos.clase import Clase
from app.modelos.progreso import ProgresoClase

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import verificar_admin, obtener_usuario_actual
from ..modelos.leccion import Leccion
from ..modelos.categoria import Categoria
from ..modelos.usuario import Usuario
from ..modelos.examen import Examen
from ..esquemas.leccion_schemas import LeccionCrear, LeccionActualizar, RespuestaAPI, RespuestaLista

router = APIRouter(prefix="/lecciones", tags=["Lecciones"])

@router.post("/", response_model=RespuestaAPI)
async def crear_leccion(
    leccion: LeccionCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        categoria_db = db.query(Categoria).filter(Categoria.id == leccion.categoria_id).first()
        if not categoria_db:
            raise HTTPException(
                status_code=400,
                detail=f"Categoría con ID {leccion.categoria_id} no existe"
            )
        
        nueva_leccion = Leccion(**leccion.dict())
        
        db.add(nueva_leccion)
        db.commit()
        db.refresh(nueva_leccion)
        
        return RespuestaAPI(
            exito=True,
            mensaje="Lección creada exitosamente",
            datos={
                "id": nueva_leccion.id,
                "titulo": nueva_leccion.titulo,
                "sena": nueva_leccion.sena,
                "categoria_id": nueva_leccion.categoria_id,
                "categoria_nombre": nueva_leccion.categoria_nombre,
                "nivel_dificultad": nueva_leccion.nivel_dificultad,
                "orden": nueva_leccion.orden,
                "puntos_base": nueva_leccion.puntos_base,
                "puntos_perfecto": nueva_leccion.puntos_perfecto,
                "activa": nueva_leccion.activa,
                "bloqueada": nueva_leccion.bloqueada,
                "total_clases": nueva_leccion.total_clases,
                "fecha_creacion": nueva_leccion.fecha_creacion
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Error al crear lección: {str(e)}"
        )

@router.get("/", response_model=RespuestaLista)
async def listar_lecciones(
    categoria_id: Optional[int] = None,
    activa: Optional[bool] = None,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        query = db.query(Leccion).options(joinedload(Leccion.categoria_rel))
        
        if activa is not None:
            query = query.filter(Leccion.activa == activa)
        
        if categoria_id:
            query = query.filter(Leccion.categoria_id == categoria_id)
        
        lecciones = query.order_by(Leccion.orden).all()
        
        datos = []
        for leccion in lecciones:
            datos.append({
                "id": leccion.id,
                "titulo": leccion.titulo,
                "descripcion": leccion.descripcion,
                "sena": leccion.sena,
                "categoria_id": leccion.categoria_id,
                "categoria_nombre": leccion.categoria_nombre,
                "orden": leccion.orden,
                "nivel_dificultad": leccion.nivel_dificultad,
                "activa": leccion.activa,
                "bloqueada": leccion.bloqueada,
                "puntos_base": leccion.puntos_base,
                "puntos_perfecto": leccion.puntos_perfecto,
                "total_clases": leccion.total_clases,
                "leccion_previa_id": leccion.leccion_previa_id,
                "fecha_creacion": leccion.fecha_creacion
            })
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(datos)} lecciones",
            datos=datos,
            total=len(datos),
            pagina=1,
            por_pagina=len(datos)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{leccion_id}", response_model=RespuestaAPI)
async def obtener_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        leccion = db.query(Leccion).options(joinedload(Leccion.categoria_rel)).filter(Leccion.id == leccion_id).first()
        if not leccion:
            raise HTTPException(status_code=404, detail="Lección no encontrada")
        
        return RespuestaAPI(
            exito=True,
            mensaje="Lección obtenida",
            datos={
                "id": leccion.id,
                "titulo": leccion.titulo,
                "descripcion": leccion.descripcion,
                "sena": leccion.sena,
                "categoria_id": leccion.categoria_id,
                "categoria_nombre": leccion.categoria_nombre,
                "orden": leccion.orden,
                "nivel_dificultad": leccion.nivel_dificultad,
                "activa": leccion.activa,
                "bloqueada": leccion.bloqueada,
                "puntos_base": leccion.puntos_base,
                "puntos_perfecto": leccion.puntos_perfecto,
                "leccion_previa_id": leccion.leccion_previa_id,
                "total_clases": leccion.total_clases,
                "fecha_creacion": leccion.fecha_creacion
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{leccion_id}", response_model=RespuestaAPI)
async def actualizar_leccion(
    leccion_id: int,
    datos: LeccionActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
        if not leccion:
            raise HTTPException(status_code=404, detail="Lección no encontrada")
        
        datos_actualizacion = datos.dict(exclude_unset=True)
        for campo, valor in datos_actualizacion.items():
            setattr(leccion, campo, valor)
        
        db.commit()
        db.refresh(leccion)
        
        return RespuestaAPI(
            exito=True,
            mensaje="Lección actualizada exitosamente",
            datos={
                "id": leccion.id,
                "titulo": leccion.titulo,
                "sena": leccion.sena,
                "categoria_id": leccion.categoria_id,
                "categoria_nombre": leccion.categoria_nombre,
                "nivel_dificultad": leccion.nivel_dificultad,
                "orden": leccion.orden,
                "puntos_base": leccion.puntos_base,
                "puntos_perfecto": leccion.puntos_perfecto,
                "activa": leccion.activa,
                "bloqueada": leccion.bloqueada,
                "total_clases": leccion.total_clases
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error al actualizar: {str(e)}"
        )

@router.delete("/{leccion_id}", response_model=RespuestaAPI)
async def eliminar_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    try:
        leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
        if not leccion:
            raise HTTPException(status_code=404, detail="Lección no encontrada")
        
        db.delete(leccion)
        db.commit()
        
        return RespuestaAPI(
            exito=True,
            mensaje="Lección eliminada exitosamente",
            datos=None
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{leccion_id}/examenes", response_model=RespuestaLista)
async def obtener_examenes_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener exámenes de una lección específica"""
    try:
        # Verificar que la lección existe
        leccion = db.query(Leccion).filter(
            Leccion.id == leccion_id,
            Leccion.activa == True
        ).first()
        
        if not leccion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        # Obtener exámenes de la lección
        examenes = db.query(Examen).filter(
            Examen.leccion_id == leccion_id,
            Examen.activo == True
        ).order_by(Examen.orden).all()
        
        # Calcular progreso del usuario en la lección
        progreso_clases = db.query(ProgresoClase).join(Clase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            Clase.leccion_id == leccion_id,
            ProgresoClase.completada == True
        ).count()
        
        total_clases = db.query(Clase).filter(
            Clase.leccion_id == leccion_id,
            Clase.activa == True
        ).count()
        
        examenes_data = []
        for examen in examenes:
            # Verificar disponibilidad del examen
            disponible = True
            
            if examen.requiere_todas_clases:
                disponible = progreso_clases >= total_clases
            elif examen.clases_requeridas > 0:
                disponible = progreso_clases >= examen.clases_requeridas
            
            # Obtener mejor resultado del usuario
            from sqlalchemy import and_, func
            from ..modelos.examen import ResultadoExamen
            
            mejor_resultado = db.query(ResultadoExamen).filter(
                and_(
                    ResultadoExamen.usuario_id == usuario_actual.id,
                    ResultadoExamen.examen_id == examen.id
                )
            ).order_by(ResultadoExamen.porcentaje.desc()).first()
            
            examen_info = {
                "id": examen.id,
                "titulo": examen.titulo,
                "descripcion": examen.descripcion,
                "tipo": examen.tipo,
                "nivel": examen.nivel,
                "leccion_id": examen.leccion_id,
                "orden": examen.orden,
                "clases_requeridas": examen.clases_requeridas,
                "requiere_todas_clases": examen.requiere_todas_clases,
                "tiempo_limite": examen.tiempo_limite,
                "puntuacion_minima": examen.puntuacion_minima,
                "activo": examen.activo,
                "total_preguntas": len(examen.preguntas) if hasattr(examen, 'preguntas') else 0,
                "completado": mejor_resultado is not None,
                "mejor_calificacion": mejor_resultado.porcentaje if mejor_resultado else 0,
                "aprobado": mejor_resultado.aprobado if mejor_resultado else False,
                "disponible": disponible,
                "fecha_creacion": examen.fecha_creacion.isoformat() if examen.fecha_creacion else None
            }
            examenes_data.append(examen_info)
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Exámenes de la lección '{leccion.titulo}'",
            datos=examenes_data,
            total=len(examenes_data),
            pagina=1,
            por_pagina=len(examenes_data)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener exámenes de lección: {str(e)}"
        )