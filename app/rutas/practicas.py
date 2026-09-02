from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.practica import Practica
from ..modelos.leccion import Leccion
from ..modelos.progreso import ProgresoLeccion
from ..esquemas.practica import PracticaCrear, PracticaRespuesta
from ..esquemas.respuestas import RespuestaAPI, RespuestaPractica

router = APIRouter(prefix="/practicas", tags=["Prácticas"])

@router.post("/", response_model=RespuestaPractica)
async def registrar_practica(
    practica: PracticaCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Registrar intento de práctica con reconocimiento IA"""
    try:
        # Obtener lección
        leccion = db.query(Leccion).filter(Leccion.id == practica.leccion_id).first()
        if not leccion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        # Calcular precisión (comparar seña esperada vs detectada)
        sena_esperada = leccion.sena.lower().strip()
        sena_detectada = practica.sena_detectada.lower().strip()
        
        # Precisión basada en coincidencia exacta y confianza
        if sena_esperada == sena_detectada:
            precision = practica.confianza  # Usar confianza del modelo
        else:
            precision = 0.0  # No coincide
        
        # Calcular puntos según precisión
        if precision >= 0.95:
            puntos = leccion.puntos_perfecto  # Puntos por perfección
            feedback = "excelente"
        elif precision >= 0.80:
            puntos = int(leccion.puntos_base * 1.5)
            feedback = "bueno"
        elif precision >= 0.60:
            puntos = leccion.puntos_base
            feedback = "regular"
        else:
            puntos = int(leccion.puntos_base * 0.5)
            feedback = "intenta_nuevamente"
        
        # Crear registro de práctica
        nueva_practica = Practica(
            usuario_id=usuario_actual.id,
            leccion_id=practica.leccion_id,
            sena_esperada=leccion.sena,
            sena_detectada=practica.sena_detectada,
            precision=precision,
            puntos_ganados=puntos,
            confianza=practica.confianza,
            tiempo_empleado=practica.tiempo_empleado,
            puntos_mano_detectados=practica.puntos_mano_detectados,
            imagen_capturada=practica.imagen_capturada,
            feedback=feedback
        )
        
        db.add(nueva_practica)
        
        # Actualizar o crear progreso
        progreso = db.query(ProgresoLeccion).filter(
            ProgresoLeccion.usuario_id == usuario_actual.id,
            ProgresoLeccion.leccion_id == practica.leccion_id
        ).first()
        
        nueva_mejor_marca = False
        
        if not progreso:
            progreso = ProgresoLeccion(
                usuario_id=usuario_actual.id,
                leccion_id=practica.leccion_id,
                mejor_precision=precision,
                total_intentos=1,
                total_puntos=puntos,
                ultima_practica=datetime.utcnow(),
                iniciada=True,
                desbloqueada=True
            )
            db.add(progreso)
            nueva_mejor_marca = True
        else:
            # Actualizar progreso
            if precision > progreso.mejor_precision:
                progreso.mejor_precision = precision
                nueva_mejor_marca = True
            
            progreso.total_intentos += 1
            progreso.total_puntos += puntos
            progreso.ultima_practica = datetime.utcnow()
            
            # Marcar como completada si alcanza 80% de precisión
            if precision >= 0.80 and not progreso.completada:
                progreso.completada = True
                progreso.fecha_completada = datetime.utcnow()
                
                # Asignar estrellas según precisión
                if precision >= 0.95:
                    progreso.estrellas = 3
                elif precision >= 0.85:
                    progreso.estrellas = 2
                else:
                    progreso.estrellas = 1
        
        db.commit()
        db.refresh(nueva_practica)
        db.refresh(progreso)
        
        # Mensaje personalizado
        if precision >= 0.95:
            mensaje = f"¡Perfecto! 🌟 {puntos} puntos"
        elif precision >= 0.80:
            mensaje = f"¡Muy bien! {puntos} puntos"
        elif precision >= 0.60:
            mensaje = f"Bien, sigue practicando. {puntos} puntos"
        else:
            mensaje = f"Intenta nuevamente. {puntos} puntos"
        
        return RespuestaPractica(
            exito=True,
            precision=precision,
            puntos_ganados=puntos,
            feedback=feedback,
            mensaje=mensaje,
            es_perfecto=precision >= 1.0,
            nueva_mejor_marca=nueva_mejor_marca,
            progreso_actualizado={
                "mejor_precision": progreso.mejor_precision,
                "total_intentos": progreso.total_intentos,
                "total_puntos": progreso.total_puntos,
                "completada": progreso.completada,
                "estrellas": progreso.estrellas
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar práctica: {str(e)}"
        )

@router.get("/leccion/{leccion_id}/historial", response_model=RespuestaAPI)
async def obtener_historial_practicas(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener historial de prácticas de una lección"""
    try:
        practicas = db.query(Practica).filter(
            Practica.usuario_id == usuario_actual.id,
            Practica.leccion_id == leccion_id
        ).order_by(Practica.fecha_practica.desc()).limit(20).all()
        
        historial = []
        for p in practicas:
            historial.append({
                "id": p.id,
                "precision": p.precision,
                "precision_porcentaje": p.precision_porcentaje,
                "puntos_ganados": p.puntos_ganados,
                "feedback": p.feedback,
                "fecha": p.fecha_practica,
                "es_perfecto": p.es_perfecto,
                "sena_esperada": p.sena_esperada,
                "sena_detectada": p.sena_detectada,
                "confianza": p.confianza,
                "tiempo_empleado": p.tiempo_empleado
            })
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Historial de {len(historial)} prácticas",
            datos=historial
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener historial: {str(e)}"
        )