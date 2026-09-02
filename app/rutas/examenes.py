from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from datetime import datetime

from app.modelos.clase import Clase
from app.modelos.progreso import ProgresoClase
from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.examen import Examen, PreguntaExamen, ResultadoExamen
from ..esquemas.examen_schemas import (
    ExamenRespuesta,
    ResultadoExamenCrear,
    ResultadoExamenRespuesta
)
from ..esquemas.respuesta_schemas import RespuestaAPI, RespuestaLista

router = APIRouter(prefix="/examenes", tags=["Exámenes"])


@router.get("/", response_model=RespuestaLista)
async def listar_examenes(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    tipo: str = None,
    nivel: int = None
):
    """Listar exámenes disponibles con resultados del usuario"""
    try:
        query = db.query(Examen).filter(Examen.activo == True)
        
        if tipo:
            query = query.filter(Examen.tipo == tipo)
        
        if nivel:
            query = query.filter(Examen.nivel == nivel)
        
        query = query.order_by(Examen.tipo, Examen.nivel)
        
        examenes = query.all()
        
        examenes_ids = [examen.id for examen in examenes]
        resultados = db.query(ResultadoExamen).filter(
            and_(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.examen_id.in_(examenes_ids)
            )
        ).all()
        
        mapa_resultados = {}
        for resultado in resultados:
            if (resultado.examen_id not in mapa_resultados or 
                resultado.fecha_finalizacion > mapa_resultados[resultado.examen_id].fecha_finalizacion):
                mapa_resultados[resultado.examen_id] = resultado
        
        examenes_data = []
        for examen in examenes:
            resultado = mapa_resultados.get(examen.id)
            
            # ✅ CORRECCIÓN: Agregar TODOS los campos necesarios del examen
            examen_info = {
                "id": examen.id,
                "titulo": examen.titulo,
                "descripcion": examen.descripcion,
                "tipo": examen.tipo,
                "nivel": examen.nivel,
                # ✅ CAMPOS CRÍTICOS AGREGADOS:
                "leccion_id": examen.leccion_id,
                "orden": examen.orden,
                "clases_requeridas": examen.clases_requeridas,
                "requiere_todas_clases": examen.requiere_todas_clases,
                "activo": examen.activo,  # ✅ IMPORTANTE: Este campo faltaba
                # Campos adicionales
                "tiempo_limite": examen.tiempo_limite,
                "puntuacion_minima": examen.puntuacion_minima,
                "total_preguntas": len(examen.preguntas) if hasattr(examen, 'preguntas') else 0,
                "fecha_creacion": examen.fecha_creacion,
                # Información del último resultado
                "ultimo_resultado": {
                    "realizado": resultado is not None,
                    "aprobado": resultado.aprobado if resultado else False,
                    "puntuacion": resultado.puntuacion_obtenida if resultado else None,
                    "porcentaje": resultado.porcentaje if resultado else None,
                    "fecha": resultado.fecha_finalizacion if resultado else None
                } if resultado else {
                    "realizado": False,
                    "aprobado": False,
                    "puntuacion": None,
                    "porcentaje": None,
                    "fecha": None
                },
                # Campos de disponibilidad (se calculan en el frontend)
                "completado": resultado is not None and resultado.aprobado,
                "disponible": None  # Se calcula en el frontend
            }
            examenes_data.append(examen_info)
        
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
            detail=f"Error al listar exámenes: {str(e)}"
        )

@router.get("/leccion/{leccion_id}", response_model=RespuestaLista)
async def obtener_examenes_por_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener exámenes asociados a una lección específica"""
    try:
        examenes = db.query(Examen).filter(
            and_(
                Examen.activo == True,
                Examen.leccion_id == leccion_id
            )
        ).order_by(Examen.orden).all()
        
        if not examenes:
            return RespuestaLista(
                exito=True,
                mensaje=f"No hay exámenes para la lección {leccion_id}",
                datos=[],
                total=0,
                pagina=1,
                por_pagina=0
            )
        
        progreso_clases = db.query(ProgresoClase).join(Clase).filter(
            and_(
                ProgresoClase.usuario_id == usuario_actual.id,
                Clase.leccion_id == leccion_id,
                ProgresoClase.completada == True
            )
        ).count()
        
        total_clases = db.query(Clase).filter(
            and_(
                Clase.leccion_id == leccion_id,
                Clase.activa == True
            )
        ).count()
        
        examenes_data = []
        for examen in examenes:
            mejor_resultado = db.query(ResultadoExamen).filter(
                and_(
                    ResultadoExamen.usuario_id == usuario_actual.id,
                    ResultadoExamen.examen_id == examen.id
                )
            ).order_by(ResultadoExamen.porcentaje.desc()).first()
            
            disponible = True
            
            if not examen.activo:
                disponible = False
            elif examen.requiere_todas_clases:
                disponible = progreso_clases >= total_clases
            elif examen.clases_requeridas > 0:
                disponible = progreso_clases >= examen.clases_requeridas
            
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
                "total_preguntas": examen.total_preguntas,
                "completado": mejor_resultado is not None,
                "mejor_calificacion": mejor_resultado.porcentaje if mejor_resultado else 0,
                "disponible": disponible
            }
            examenes_data.append(examen_info)
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Exámenes de la lección {leccion_id}",
            datos=examenes_data,
            total=len(examenes_data),
            pagina=1,
            por_pagina=len(examenes_data)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener exámenes de lección: {str(e)}"
        )


@router.get("/leccion/{leccion_id}/progreso", response_model=RespuestaAPI)
async def obtener_progreso_examenes_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener progreso de exámenes para una lección específica"""
    try:
        examenes = db.query(Examen).filter(
            and_(
                Examen.leccion_id == leccion_id,
                Examen.activo == True
            )
        ).all()
        
        progreso_examenes = []
        for examen in examenes:
            mejor_resultado = db.query(ResultadoExamen).filter(
                and_(
                    ResultadoExamen.examen_id == examen.id,
                    ResultadoExamen.usuario_id == usuario_actual.id,
                    ResultadoExamen.aprobado == True
                )
            ).order_by(ResultadoExamen.porcentaje.desc()).first()
            
            total_intentos = db.query(ResultadoExamen).filter(
                and_(
                    ResultadoExamen.examen_id == examen.id,
                    ResultadoExamen.usuario_id == usuario_actual.id
                )
            ).count()
            
            ultimo_intento = db.query(ResultadoExamen).filter(
                and_(
                    ResultadoExamen.examen_id == examen.id,
                    ResultadoExamen.usuario_id == usuario_actual.id
                )
            ).order_by(ResultadoExamen.fecha_finalizacion.desc()).first()
            
            progreso_examenes.append({
                "examen_id": examen.id,
                "titulo": examen.titulo,
                "completado": mejor_resultado is not None,
                "mejor_calificacion": mejor_resultado.porcentaje if mejor_resultado else 0,
                "intentos": total_intentos,
                "ultimo_intento": ultimo_intento.fecha_finalizacion if ultimo_intento else None,
                "aprobado": mejor_resultado.aprobado if mejor_resultado else False
            })
        
        total_examenes = len(progreso_examenes)
        examenes_completados = sum(1 for p in progreso_examenes if p["completado"])
        promedio_calificacion = sum(p["mejor_calificacion"] for p in progreso_examenes) / total_examenes if total_examenes > 0 else 0
        
        return RespuestaAPI(
            exito=True,
            mensaje="Progreso de exámenes obtenido",
            datos={
                "progreso_examenes": progreso_examenes,
                "resumen": {
                    "total_examenes": total_examenes,
                    "examenes_completados": examenes_completados,
                    "examenes_pendientes": total_examenes - examenes_completados,
                    "promedio_calificacion": round(promedio_calificacion, 1),
                    "porcentaje_completado": round((examenes_completados / total_examenes * 100), 1) if total_examenes > 0 else 0
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener progreso de exámenes: {str(e)}"
        )


@router.get("/{examen_id}", response_model=RespuestaAPI)
async def obtener_examen(
    examen_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener un examen específico con sus preguntas"""
    try:
        examen = db.query(Examen).filter(
            and_(
                Examen.id == examen_id,
                Examen.activo == True
            )
        ).first()
        
        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )
        
        preguntas = db.query(PreguntaExamen).filter(
            PreguntaExamen.examen_id == examen_id
        ).order_by(PreguntaExamen.orden).all()
        
        preguntas_data = []
        for pregunta in preguntas:
            pregunta_info = {
                "id": pregunta.id,
                "pregunta": pregunta.pregunta,
                "tipo_pregunta": pregunta.tipo_pregunta,
                "opciones": pregunta.opciones if pregunta.opciones else None,
                "sena_esperada": pregunta.sena_esperada if pregunta.tipo_pregunta == "reconocimiento" else None,
                "imagen_sena": pregunta.imagen_sena,
                "puntos": pregunta.puntos,
                "orden": pregunta.orden
            }
            preguntas_data.append(pregunta_info)
        
        ultimo_resultado = db.query(ResultadoExamen).filter(
            and_(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.examen_id == examen_id
            )
        ).order_by(ResultadoExamen.fecha_finalizacion.desc()).first()
        
        examen_data = {
            "id": examen.id,
            "titulo": examen.titulo,
            "descripcion": examen.descripcion,
            "tipo": examen.tipo,
            "nivel": examen.nivel,
            "tiempo_limite": examen.tiempo_limite,
            "puntuacion_minima": examen.puntuacion_minima,
            "total_preguntas": len(preguntas_data),
            "puntos_totales": sum(p.puntos for p in preguntas),
            "preguntas": preguntas_data,
            "ultimo_resultado": {
                "realizado": ultimo_resultado is not None,
                "aprobado": ultimo_resultado.aprobado if ultimo_resultado else False,
                "puntuacion": ultimo_resultado.puntuacion_obtenida if ultimo_resultado else None,
                "porcentaje": ultimo_resultado.porcentaje if ultimo_resultado else None,
                "fecha": ultimo_resultado.fecha_finalizacion if ultimo_resultado else None
            } if ultimo_resultado else None
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Examen encontrado",
            datos=examen_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener examen: {str(e)}"
        )


@router.post("/{examen_id}/presentar", response_model=RespuestaAPI)
async def presentar_examen(
    examen_id: int,
    data: Dict[str, Any],  # ✅ Recibir todo el body como dict
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Presentar un examen con las respuestas del usuario"""
    try:
        # ✅ Extraer tiempo_empleado del body
        tiempo_empleado = data.pop('tiempo_empleado', None)
        respuestas = data  # El resto son las respuestas
        
        print(f"\n{'='*60}")
        print(f"📝 PRESENTANDO EXAMEN {examen_id}")
        print(f"👤 Usuario: {usuario_actual.id}")
        print(f"📥 Body completo: {data}")
        print(f"📝 Respuestas: {respuestas}")
        print(f"⏱️  Tiempo empleado: {tiempo_empleado} segundos")
        print(f"{'='*60}\n")
        
        examen = db.query(Examen).filter(
            and_(
                Examen.id == examen_id,
                Examen.activo == True
            )
        ).first()
        
        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )
        
        preguntas = db.query(PreguntaExamen).filter(
            PreguntaExamen.examen_id == examen_id
        ).all()
        
        print(f"📋 Total de preguntas en el examen: {len(preguntas)}")
        
        if not preguntas:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El examen no tiene preguntas configuradas"
            )
        
        puntuacion_obtenida = 0
        puntuacion_maxima = sum(pregunta.puntos for pregunta in preguntas)
        respuestas_detalladas = {}
        
        print(f"\n🎯 CALIFICANDO RESPUESTAS:")
        print(f"{'─'*60}")
        
        for pregunta in preguntas:
            pregunta_id_str = str(pregunta.id)
            respuesta_usuario = respuestas.get(pregunta_id_str)
            
            es_correcta = False
            
            if pregunta.tipo_pregunta == "multiple":
                es_correcta = respuesta_usuario == pregunta.respuesta_correcta
                
            elif pregunta.tipo_pregunta == "verdadero_falso":
                es_correcta = respuesta_usuario == pregunta.respuesta_correcta
                
            elif pregunta.tipo_pregunta == "reconocimiento":
                # ✅ CORRECCIÓN: Normalizar strings para comparación
                respuesta_normalizada = str(respuesta_usuario).upper().strip() if respuesta_usuario else ""
                sena_esperada_normalizada = str(pregunta.sena_esperada).upper().strip() if pregunta.sena_esperada else ""
                
                print(f"   🔍 Pregunta {pregunta.id} (reconocimiento):")
                print(f"      Respuesta usuario: '{respuesta_normalizada}'")
                print(f"      Seña esperada: '{sena_esperada_normalizada}'")
                
                # ✅ Validar que sena_esperada no sea NULL o vacía
                if not sena_esperada_normalizada:
                    print(f"      ⚠️  ERROR: sena_esperada es NULL o vacía")
                    # Si no hay seña esperada configurada, buscar en la lección
                    from app.modelos.leccion import Leccion
                    if pregunta.leccion_id:
                        leccion = db.query(Leccion).filter(Leccion.id == pregunta.leccion_id).first()
                        if leccion and leccion.sena:
                            sena_esperada_normalizada = leccion.sena.upper().strip()
                            print(f"      🔄 Usando seña de la lección: '{sena_esperada_normalizada}'")
                
                es_correcta = respuesta_normalizada == sena_esperada_normalizada and len(sena_esperada_normalizada) > 0
                print(f"      {'✅' if es_correcta else '❌'} Correcta: {es_correcta}")
            
            puntos_pregunta = pregunta.puntos if es_correcta else 0
            puntuacion_obtenida += puntos_pregunta
            
            print(f"      💯 Puntos: {puntos_pregunta}/{pregunta.puntos}")
            print(f"{'─'*60}")
            
            respuestas_detalladas[pregunta_id_str] = {
                "respuesta_usuario": respuesta_usuario,
                "respuesta_correcta": pregunta.respuesta_correcta or pregunta.sena_esperada,
                "es_correcta": es_correcta,
                "puntos_obtenidos": puntos_pregunta,
                "puntos_maximos": pregunta.puntos
            }
        
        print(f"\n📊 RESULTADOS FINALES:")
        print(f"   Puntuación obtenida: {puntuacion_obtenida}/{puntuacion_maxima}")
        
        porcentaje = (puntuacion_obtenida / puntuacion_maxima * 100) if puntuacion_maxima > 0 else 0
        print(f"   Porcentaje: {porcentaje:.1f}%")
        print(f"   Puntuación mínima: {examen.puntuacion_minima}%")
        
        aprobado = porcentaje >= examen.puntuacion_minima
        print(f"   {'✅ APROBADO' if aprobado else '❌ NO APROBADO'}")
        print(f"{'='*60}\n")
        
        resultado = ResultadoExamen(
            usuario_id=usuario_actual.id,
            examen_id=examen_id,
            puntuacion_obtenida=puntuacion_obtenida,
            puntuacion_maxima=puntuacion_maxima,
            porcentaje=porcentaje,
            aprobado=aprobado,
            tiempo_empleado=tiempo_empleado,
            respuestas=respuestas_detalladas,
            fecha_inicio=datetime.utcnow(),
            fecha_finalizacion=datetime.utcnow()
        )
        
        db.add(resultado)
        db.commit()
        db.refresh(resultado)
        
        return RespuestaAPI(
            exito=True,
            mensaje="Examen presentado exitosamente" + (" - ¡APROBADO!" if aprobado else " - No aprobado"),
            datos={
                "resultado_id": resultado.id,
                "examen_titulo": examen.titulo,
                "puntuacion_obtenida": puntuacion_obtenida,
                "puntuacion_maxima": puntuacion_maxima,
                "porcentaje": round(porcentaje, 1),
                "puntuacion_minima": examen.puntuacion_minima,
                "aprobado": aprobado,
                "tiempo_empleado": tiempo_empleado,
                "respuestas_correctas": sum(1 for r in respuestas_detalladas.values() if r["es_correcta"]),
                "total_preguntas": len(preguntas),
                "fecha_finalizacion": resultado.fecha_finalizacion
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al presentar examen: {str(e)}"
        )


@router.get("/{examen_id}/resultados", response_model=RespuestaLista)
async def obtener_resultados_examen(
    examen_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener todos los resultados del usuario para un examen específico"""
    try:
        examen = db.query(Examen).filter(Examen.id == examen_id).first()
        if not examen:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Examen no encontrado"
            )
        
        resultados = db.query(ResultadoExamen).filter(
            and_(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.examen_id == examen_id
            )
        ).order_by(ResultadoExamen.fecha_finalizacion.desc()).all()
        
        resultados_data = []
        for resultado in resultados:
            resultado_info = {
                "id": resultado.id,
                "puntuacion_obtenida": resultado.puntuacion_obtenida,
                "puntuacion_maxima": resultado.puntuacion_maxima,
                "porcentaje": resultado.porcentaje,
                "aprobado": resultado.aprobado,
                "tiempo_empleado": resultado.tiempo_empleado,
                "fecha_finalizacion": resultado.fecha_finalizacion,
                "respuestas_correctas": sum(1 for r in resultado.respuestas.values() if r.get("es_correcta", False)) if resultado.respuestas else 0,
                "total_preguntas": len(resultado.respuestas) if resultado.respuestas else 0
            }
            resultados_data.append(resultado_info)
        
        mejor_resultado = max(resultados, key=lambda x: x.porcentaje) if resultados else None
        promedio_porcentaje = sum(r.porcentaje for r in resultados) / len(resultados) if resultados else 0
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Historial de {len(resultados_data)} intentos",
            datos=resultados_data,
            total=len(resultados_data),
            pagina=1,
            por_pagina=len(resultados_data)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener resultados: {str(e)}"
        )