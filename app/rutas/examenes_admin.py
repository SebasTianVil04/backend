from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import verificar_admin
from ..modelos.examen import Examen, PreguntaExamen, ResultadoExamen
from ..modelos.usuario import Usuario
from ..modelos.leccion import Leccion
from ..esquemas.respuesta_schemas import RespuestaAPI, RespuestaLista

router = APIRouter(prefix="", tags=["Admin - Exámenes"])
logger = logging.getLogger(__name__)


class ExamenCrear(BaseModel):
    titulo: str
    descripcion: Optional[str] = None
    tipo: str = 'nivel'
    nivel: Optional[int] = None
    leccion_id: int
    orden: Optional[int] = 1
    clases_requeridas: Optional[int] = 0
    requiere_todas_clases: Optional[bool] = False
    tiempo_limite: Optional[int] = None
    puntuacion_minima: float = 70.0


class ExamenActualizar(BaseModel):
    titulo: Optional[str] = None
    descripcion: Optional[str] = None
    tipo: Optional[str] = None
    nivel: Optional[int] = None
    leccion_id: Optional[int] = None
    orden: Optional[int] = None
    clases_requeridas: Optional[int] = None
    requiere_todas_clases: Optional[bool] = None
    tiempo_limite: Optional[int] = None
    puntuacion_minima: Optional[float] = None
    activo: Optional[bool] = None


class PreguntaCrear(BaseModel):
    examen_id: int
    leccion_id: Optional[int] = None
    pregunta: str
    tipo_pregunta: str
    sena_esperada: Optional[str] = None
    imagen_sena: Optional[str] = None
    opciones: Optional[dict] = None
    respuesta_correcta: Optional[str] = None
    puntos: int = 10
    orden: int


class PreguntaActualizar(BaseModel):
    pregunta: Optional[str] = None
    sena_esperada: Optional[str] = None
    imagen_sena: Optional[str] = None
    opciones: Optional[dict] = None
    respuesta_correcta: Optional[str] = None
    puntos: Optional[int] = None
    orden: Optional[int] = None


@router.post("", response_model=RespuestaAPI)
async def crear_examen(
    examen: ExamenCrear,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Crear un nuevo examen"""
    try:
        if examen.tipo == 'nivel' and examen.nivel is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nivel es obligatorio para exámenes de nivel"
            )

        if examen.tipo == 'final' and examen.nivel is not None:
            examen.nivel = None

        leccion_existente = db.query(Leccion).filter(Leccion.id == examen.leccion_id).first()
        if not leccion_existente:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"La lección con ID {examen.leccion_id} no existe"
            )

        nuevo_examen = Examen(
            titulo=examen.titulo,
            descripcion=examen.descripcion,
            tipo=examen.tipo,
            nivel=examen.nivel,
            leccion_id=examen.leccion_id,
            orden=examen.orden or 1,
            clases_requeridas=examen.clases_requeridas or 0,
            requiere_todas_clases=examen.requiere_todas_clases or False,
            tiempo_limite=examen.tiempo_limite,
            puntuacion_minima=examen.puntuacion_minima,
            activo=True
        )

        db.add(nuevo_examen)
        db.commit()
        db.refresh(nuevo_examen)

        return RespuestaAPI(
            exito=True,
            mensaje="Examen creado exitosamente",
            datos={
                "id": nuevo_examen.id,
                "titulo": nuevo_examen.titulo,
                "tipo": nuevo_examen.tipo,
                "nivel": nuevo_examen.nivel,
                "leccion_id": nuevo_examen.leccion_id
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear examen: {str(e)}"
        )


@router.get("", response_model=RespuestaLista)
async def listar_examenes_admin(
    tipo: Optional[str] = None,
    activo: Optional[bool] = None,
    leccion_id: Optional[int] = None,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Listar todos los exámenes (admin)"""
    try:
        query = db.query(Examen)

        if tipo:
            query = query.filter(Examen.tipo == tipo)
        if activo is not None:
            query = query.filter(Examen.activo == activo)
        if leccion_id is not None:
            query = query.filter(Examen.leccion_id == leccion_id)

        examenes = query.order_by(Examen.leccion_id, Examen.orden).all()

        examenes_data = []
        for examen in examenes:
            examenes_data.append({
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
                "total_preguntas": len(examen.preguntas),
                "fecha_creacion": examen.fecha_creacion
            })

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(examenes_data)} exámenes",
            datos=examenes_data,
            total=len(examenes_data),
            pagina=1,
            por_pagina=len(examenes_data)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{examen_id}", response_model=RespuestaAPI)
async def obtener_examen_admin(
    examen_id: int,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Obtener detalles completos de un examen"""
    examen = db.query(Examen).filter(Examen.id == examen_id).first()

    if not examen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Examen no encontrado"
        )

    preguntas = db.query(PreguntaExamen).filter(
        PreguntaExamen.examen_id == examen_id
    ).order_by(PreguntaExamen.orden).all()

    preguntas_data = []
    for p in preguntas:
        preguntas_data.append({
            "id": p.id,
            "pregunta": p.pregunta,
            "tipo_pregunta": p.tipo_pregunta,
            "sena_esperada": p.sena_esperada,
            "imagen_sena": p.imagen_sena,
            "opciones": p.opciones,
            "respuesta_correcta": p.respuesta_correcta,
            "puntos": p.puntos,
            "orden": p.orden,
            "leccion_id": p.leccion_id
        })

    return RespuestaAPI(
        exito=True,
        mensaje="Examen encontrado",
        datos={
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
            "preguntas": preguntas_data,
            "total_preguntas": len(preguntas_data),
            "puntos_totales": sum(p.puntos for p in preguntas)
        }
    )


@router.put("/{examen_id}", response_model=RespuestaAPI)
async def actualizar_examen(
    examen_id: int,
    datos: ExamenActualizar,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Actualizar un examen"""
    try:
        examen = db.query(Examen).filter(Examen.id == examen_id).first()

        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )

        if datos.tipo is not None:
            if datos.tipo == 'final':
                datos.nivel = None
            elif datos.tipo == 'nivel' and datos.nivel is None and examen.nivel is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El nivel es obligatorio para exámenes de nivel"
                )

        if datos.leccion_id is not None:
            leccion_existente = db.query(Leccion).filter(Leccion.id == datos.leccion_id).first()
            if not leccion_existente:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"La lección con ID {datos.leccion_id} no existe"
                )

        campos_actualizados = {}
        for campo, valor in datos.dict(exclude_unset=True).items():
            setattr(examen, campo, valor)
            campos_actualizados[campo] = valor

        db.commit()
        db.refresh(examen)

        return RespuestaAPI(
            exito=True,
            mensaje="Examen actualizado exitosamente",
            datos={
                "id": examen.id,
                "titulo": examen.titulo,
                "tipo": examen.tipo,
                "nivel": examen.nivel,
                "leccion_id": examen.leccion_id,
                **campos_actualizados
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar examen: {str(e)}"
        )


@router.delete("/{examen_id}", response_model=RespuestaAPI)
async def eliminar_examen(
    examen_id: int,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Eliminar un examen"""
    try:
        examen = db.query(Examen).filter(Examen.id == examen_id).first()

        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )

        db.delete(examen)
        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje="Examen eliminado exitosamente"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar examen: {str(e)}"
        )


# ===== GESTIÓN DE PREGUNTAS =====

@router.post("/preguntas", response_model=RespuestaAPI)
async def crear_pregunta(
    pregunta: PreguntaCrear,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Crear una nueva pregunta para un examen"""
    try:
        examen = db.query(Examen).filter(Examen.id == pregunta.examen_id).first()
        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )

        nueva_pregunta = PreguntaExamen(**pregunta.dict())
        db.add(nueva_pregunta)
        db.commit()
        db.refresh(nueva_pregunta)

        return RespuestaAPI(
            exito=True,
            mensaje="Pregunta creada exitosamente",
            datos={"id": nueva_pregunta.id}
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear pregunta: {str(e)}"
        )


@router.put("/preguntas/{pregunta_id}", response_model=RespuestaAPI)
async def actualizar_pregunta(
    pregunta_id: int,
    datos: PreguntaActualizar,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Actualizar una pregunta"""
    try:
        pregunta = db.query(PreguntaExamen).filter(
            PreguntaExamen.id == pregunta_id
        ).first()

        if not pregunta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pregunta no encontrada"
            )

        for campo, valor in datos.dict(exclude_unset=True).items():
            setattr(pregunta, campo, valor)

        db.commit()
        db.refresh(pregunta)

        return RespuestaAPI(
            exito=True,
            mensaje="Pregunta actualizada exitosamente",
            datos={"id": pregunta.id}
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar pregunta: {str(e)}"
        )


@router.delete("/preguntas/{pregunta_id}", response_model=RespuestaAPI)
async def eliminar_pregunta(
    pregunta_id: int,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Eliminar una pregunta"""
    try:
        pregunta = db.query(PreguntaExamen).filter(
            PreguntaExamen.id == pregunta_id
        ).first()

        if not pregunta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pregunta no encontrada"
            )

        db.delete(pregunta)
        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje="Pregunta eliminada exitosamente"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar pregunta: {str(e)}"
        )


@router.get("/{examen_id}/preguntas", response_model=RespuestaLista)
async def listar_preguntas_examen(
    examen_id: int,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Listar todas las preguntas de un examen"""
    try:
        preguntas = db.query(PreguntaExamen).filter(
            PreguntaExamen.examen_id == examen_id
        ).order_by(PreguntaExamen.orden).all()

        preguntas_data = []
        for pregunta in preguntas:
            preguntas_data.append({
                "id": pregunta.id,
                "examen_id": pregunta.examen_id,
                "leccion_id": pregunta.leccion_id,
                "pregunta": pregunta.pregunta,
                "tipo_pregunta": pregunta.tipo_pregunta,
                "sena_esperada": pregunta.sena_esperada,
                "imagen_sena": pregunta.imagen_sena,
                "opciones": pregunta.opciones,
                "respuesta_correcta": pregunta.respuesta_correcta,
                "puntos": pregunta.puntos,
                "orden": pregunta.orden,
                "fecha_creacion": pregunta.fecha_creacion
            })

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(preguntas_data)} preguntas",
            datos=preguntas_data,
            total=len(preguntas_data),
            pagina=1,
            por_pagina=len(preguntas_data)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar preguntas: {str(e)}"
        )


@router.get("/{examen_id}/estadisticas", response_model=RespuestaAPI)
async def obtener_estadisticas_examen(
    examen_id: int,
    db: Session = Depends(obtener_bd),
    admin = Depends(verificar_admin)
):
    """Obtener estadísticas reales de los intentos de examen"""
    try:
        examen = db.query(Examen).filter(Examen.id == examen_id).first()
        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )

        resultados = db.query(ResultadoExamen).filter(
            ResultadoExamen.examen_id == examen_id
        ).all()

        total_intentos = len(resultados)

        if total_intentos == 0:
            return RespuestaAPI(
                exito=True,
                mensaje="No hay intentos registrados para este examen",
                datos={
                    "total_intentos": 0,
                    "promedio_puntuacion": 0,
                    "tasa_aprobacion": 0,
                    "intentos_recientes": [],
                    "distribucion_puntuaciones": [],
                    "progreso_tiempo": []
                }
            )

        porcentajes = [r.porcentaje for r in resultados if r.porcentaje is not None]
        promedio_puntuacion = sum(porcentajes) / len(porcentajes) if porcentajes else 0

        examenes_aprobados = sum(1 for r in resultados if r.aprobado)
        tasa_aprobacion = (examenes_aprobados / total_intentos) * 100

        intentos_recientes = db.query(
            ResultadoExamen,
            Usuario
        ).join(
            Usuario, ResultadoExamen.usuario_id == Usuario.id
        ).filter(
            ResultadoExamen.examen_id == examen_id
        ).order_by(
            ResultadoExamen.fecha_finalizacion.desc()
        ).limit(10).all()

        intentos_recientes_data = []
        for resultado, usuario in intentos_recientes:
            intentos_recientes_data.append({
                "usuario_id": usuario.id,
                "usuario_nombre": usuario.nombre_completo,
                "usuario_email": usuario.email,
                "puntuacion": resultado.porcentaje,
                "puntuacion_obtenida": resultado.puntuacion_obtenida,
                "puntuacion_maxima": resultado.puntuacion_maxima,
                "aprobado": resultado.aprobado,
                "fecha_intento": resultado.fecha_finalizacion.isoformat() if resultado.fecha_finalizacion else None,
                "duracion_segundos": resultado.tiempo_empleado or 0
            })

        distribucion = []
        rangos = [
            ("0-20", 0, 20),
            ("21-40", 21, 40),
            ("41-60", 41, 60),
            ("61-80", 61, 80),
            ("81-100", 81, 100)
        ]

        for rango_nombre, min_val, max_val in rangos:
            cantidad = sum(
                1 for r in resultados
                if r.porcentaje is not None and min_val <= r.porcentaje <= max_val
            )
            distribucion.append({
                "rango": rango_nombre,
                "cantidad": cantidad
            })

        fecha_limite = datetime.utcnow() - timedelta(days=30)
        progreso_tiempo = db.query(
            func.date(ResultadoExamen.fecha_finalizacion).label('fecha'),
            func.count(ResultadoExamen.id).label('intentos'),
            func.avg(ResultadoExamen.porcentaje).label('promedio_puntuacion')
        ).filter(
            ResultadoExamen.examen_id == examen_id,
            ResultadoExamen.fecha_finalizacion >= fecha_limite,
            ResultadoExamen.porcentaje.isnot(None)
        ).group_by(
            func.date(ResultadoExamen.fecha_finalizacion)
        ).order_by(
            func.date(ResultadoExamen.fecha_finalizacion)
        ).all()

        progreso_tiempo_data = []
        for pt in progreso_tiempo:
            progreso_tiempo_data.append({
                "fecha": pt.fecha.isoformat(),
                "intentos": pt.intentos,
                "promedio_puntuacion": float(pt.promedio_puntuacion or 0)
            })

        estadisticas = {
            "total_intentos": total_intentos,
            "promedio_puntuacion": round(promedio_puntuacion, 2),
            "tasa_aprobacion": round(tasa_aprobacion, 2),
            "intentos_recientes": intentos_recientes_data,
            "distribucion_puntuaciones": distribucion,
            "progreso_tiempo": progreso_tiempo_data,
            "detalles_examen": {
                "puntuacion_minima": examen.puntuacion_minima,
                "total_preguntas": examen.total_preguntas,
                "puntos_totales": examen.puntos_totales
            }
        }

        return RespuestaAPI(
            exito=True,
            mensaje=f"Estadísticas del examen '{examen.titulo}'",
            datos=estadisticas
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del examen {examen_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )