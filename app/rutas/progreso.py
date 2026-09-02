from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy import and_  
from app.modelos.examen import Examen, ResultadoExamen

from ..utilidades.base_datos import obtener_bd
from ..modelos.progreso import ProgresoClase, ProgresoLeccion
from ..modelos.clase import Clase
from ..modelos.leccion import Leccion
from ..modelos.usuario import Usuario
from ..esquemas.progreso_schemas import (
    ProgresoClaseRespuesta, 
    ProgresoLeccionRespuesta,
    RegistroIntentoRequest,
    RegistroIntentoRespuesta,
    ResumenDesempenoClase,
    ResultadoPracticaRequest,
    RequisitosClase,
    RespuestaGuardarPractica
)
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista, RespuestaPractica
from ..utilidades.seguridad import obtener_usuario_actual

router = APIRouter(prefix="/progreso", tags=["Progreso"])

@router.get("/clase/{clase_id}", response_model=ProgresoClaseRespuesta)
def obtener_progreso_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    progreso = db.query(ProgresoClase).filter(
        ProgresoClase.usuario_id == usuario_actual.id,
        ProgresoClase.clase_id == clase_id
    ).first()
    
    if not progreso:
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        if not clase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clase no encontrada"
            )
        
        progreso = ProgresoClase(
            usuario_id=usuario_actual.id,
            clase_id=clase_id,
            vista=False,
            completada=False,
            aprobada=False,
            intentos_realizados=0,
            mejor_precision=0.0,
            tiempo_total_practica=0
        )
        db.add(progreso)
        db.commit()
        db.refresh(progreso)
    
    return progreso

@router.post("/clase/{clase_id}/marcar-vista")
def marcar_clase_vista(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    progreso = db.query(ProgresoClase).filter(
        ProgresoClase.usuario_id == usuario_actual.id,
        ProgresoClase.clase_id == clase_id
    ).first()
    
    if not progreso:
        progreso = ProgresoClase(
            usuario_id=usuario_actual.id,
            clase_id=clase_id,
            vista=False,
            completada=False,
            aprobada=False,
            intentos_realizados=0,
            mejor_precision=0.0,
            tiempo_total_practica=0
        )
        db.add(progreso)
    
    if not progreso.vista:
        progreso.vista = True
        progreso.fecha_primera_vista = datetime.now()
    
    db.commit()
    db.refresh(progreso)
    
    return {"mensaje": "Clase marcada como vista", "progreso": progreso}

@router.post("/clases/{clase_id}/practica", response_model=RespuestaAPI)
async def guardar_resultado_practica(
    clase_id: int,
    datos: RegistroIntentoRequest,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd)
):
    try:
        # Obtener o crear progreso de la clase
        progreso = db.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.clase_id == clase_id
        ).first()
        
        if not progreso:
            # Crear nuevo progreso si no existe
            progreso = ProgresoClase(
                usuario_id=usuario_actual.id,
                clase_id=clase_id,
                vista=True
            )
            db.add(progreso)
            db.flush()
        
        # Obtener la clase para los requisitos
        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        if not clase:
            raise HTTPException(status_code=404, detail="Clase no encontrada")
        
        # Registrar el intento
        resultado = progreso.registrar_intento(
            precision=datos.precision,
            duracion_segundos=datos.tiempo_practica,
            clase_obj=clase,
            es_exitoso=datos.es_exitoso
        )
        
        # Guardar cambios
        db.commit()
        db.refresh(progreso)
        
        # Preparar respuesta - CONVERTIR a dict explícitamente
        respuesta_dict = {
            'puntos_ganados': resultado['puntos_ganados'],
            'xp_ganado': resultado['xp_ganado'],
            'precision': resultado['precision'],
            'es_exitoso': resultado['es_exitoso'],
            'nivel_dominio': resultado['nivel_dominio'],
            'clase_aprobada': progreso.aprobada,
            'clase_completada': progreso.completada
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Práctica registrada exitosamente",
            datos=respuesta_dict  # Asegurar que es un diccionario
        )
        
    except Exception as e:
        db.rollback()
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error al guardar práctica: {str(e)}",
            errores=[str(e)]
        )
@router.get("/clase/{clase_id}/resumen", response_model=ResumenDesempenoClase)
def obtener_resumen_desempeno_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    progreso = db.query(ProgresoClase).filter(
        ProgresoClase.usuario_id == usuario_actual.id,
        ProgresoClase.clase_id == clase_id
    ).first()
    
    if not progreso:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progreso no encontrado para esta clase"
        )
    
    resumen = {
        "clase_id": clase_id,
        "completada": progreso.completada,
        "aprobada": progreso.aprobada,
        "intentos_realizados": progreso.intentos_realizados,
        "intentos_exitosos": progreso.intentos_exitosos,
        "tasa_exito": progreso.tasa_exito,
        "mejor_precision": progreso.mejor_precision,
        "precision_promedio": progreso.precision_promedio,
        "ultima_precision": progreso.ultima_precision,
        "nivel_dominio": progreso.nivel_dominio,
        "puntos_ganados": progreso.puntos_ganados,
        "xp_ganado": progreso.xp_ganado,
        "tiempo_total_practica_minutos": progreso.tiempo_total_practica // 60,
        "fecha_completada": progreso.fecha_completada.isoformat() if progreso.fecha_completada else None
    }
    
    return ResumenDesempenoClase(**resumen)

@router.get("/leccion/{leccion_id}", response_model=ProgresoLeccionRespuesta)
def obtener_progreso_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    progreso = db.query(ProgresoLeccion).filter(
        ProgresoLeccion.usuario_id == usuario_actual.id,
        ProgresoLeccion.leccion_id == leccion_id
    ).first()
    
    if not progreso:
        leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
        if not leccion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lección no encontrada"
            )
        
        progreso = ProgresoLeccion(
            usuario_id=usuario_actual.id,
            leccion_id=leccion_id,
            total_clases=len(leccion.clases) if leccion.clases else 0,
            desbloqueada=True if leccion.orden == 1 else False
        )
        db.add(progreso)
        db.commit()
        db.refresh(progreso)
    
    return progreso

@router.get("/usuario/resumen")
def obtener_resumen_progreso_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        lecciones = db.query(Leccion).filter(Leccion.activa == True).all()
        
        progresos_data = []
        total_puntos = 0
        total_estrellas = 0
        lecciones_completadas = 0
        
        total_clases_global = 0
        clases_completadas_global = 0
        
        puntos_examenes = db.query(func.sum(ResultadoExamen.puntuacion_obtenida)).filter(
            ResultadoExamen.usuario_id == usuario_actual.id,
            ResultadoExamen.aprobado == True
        ).scalar() or 0
        
        for leccion in lecciones:
            progreso = db.query(ProgresoLeccion).filter(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.leccion_id == leccion.id
            ).first()
            
            stats_clases = db.query(
                func.sum(ProgresoClase.puntos_ganados).label('total_puntos_clases'),
                func.sum(ProgresoClase.xp_ganado).label('total_xp_clases'),
                func.count(ProgresoClase.id).label('total_intentos'),
                func.avg(ProgresoClase.mejor_precision).label('precision_promedio')
            ).join(Clase).filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                Clase.leccion_id == leccion.id,
                Clase.activa == True
            ).first()
            
            clases_completadas_count = db.query(ProgresoClase).join(Clase).filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                Clase.leccion_id == leccion.id,
                ProgresoClase.completada == True,
                Clase.activa == True
            ).count()
            
            total_clases = db.query(Clase).filter(
                Clase.leccion_id == leccion.id,
                Clase.activa == True
            ).count()
            
            total_clases_global += total_clases
            clases_completadas_global += clases_completadas_count
            
            puntos_examenes_leccion = db.query(func.sum(ResultadoExamen.puntuacion_obtenida)).join(Examen).filter(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.aprobado == True,
                Examen.leccion_id == leccion.id
            ).scalar() or 0
            
            porcentaje_completado = (clases_completadas_count / total_clases * 100) if total_clases > 0 else 0
            completada_actual = clases_completadas_count >= total_clases and total_clases > 0

            if not progreso:
                progreso = ProgresoLeccion(
                    usuario_id=usuario_actual.id,
                    leccion_id=leccion.id,
                    total_clases=total_clases,
                    clases_completadas=clases_completadas_count,
                    desbloqueada=True if leccion.orden == 1 else False,
                    iniciada=clases_completadas_count > 0,
                    completada=completada_actual,
                    mejor_precision=float(stats_clases.precision_promedio or 0),
                    total_intentos=stats_clases.total_intentos or 0,
                    total_puntos=int((stats_clases.total_puntos_clases or 0) + puntos_examenes_leccion),
                    xp_total=int(stats_clases.total_xp_clases or 0),
                    estrellas=0
                )
                db.add(progreso)
                db.flush()
            else:
                progreso.clases_completadas = clases_completadas_count
                progreso.total_clases = total_clases
                progreso.total_puntos = int((stats_clases.total_puntos_clases or 0) + puntos_examenes_leccion)
                progreso.xp_total = int(stats_clases.total_xp_clases or 0)
                progreso.total_intentos = stats_clases.total_intentos or 0
                
                if clases_completadas_count > 0 and not progreso.iniciada:
                    progreso.iniciada = True
                    progreso.fecha_inicio = datetime.now()
                
                if completada_actual and not progreso.completada:
                    progreso.completada = True
                    progreso.fecha_completada = datetime.now()
                elif not completada_actual and progreso.completada:
                    progreso.completada = False
                    progreso.fecha_completada = None
                
                if stats_clases.precision_promedio:
                    progreso.mejor_precision = float(stats_clases.precision_promedio)
                    
                    if progreso.mejor_precision >= 0.95:
                        progreso.estrellas = 3
                    elif progreso.mejor_precision >= 0.85:
                        progreso.estrellas = 2
                    else:
                        progreso.estrellas = 1
            
            progresos_data.append({
                "id": progreso.id if progreso.id else None,
                "leccion_id": leccion.id,
                "leccion_titulo": leccion.titulo,
                "usuario_id": usuario_actual.id,
                "desbloqueada": progreso.desbloqueada,
                "iniciada": progreso.iniciada,
                "completada": completada_actual,
                "total_clases": total_clases,
                "clases_completadas": clases_completadas_count,
                "mejor_precision": float(progreso.mejor_precision or 0),
                "total_intentos": progreso.total_intentos or 0,
                "total_puntos": progreso.total_puntos or 0,
                "estrellas": progreso.estrellas or 0,
                "porcentaje_completado": round(porcentaje_completado, 1),
                "porcentaje_precision": f"{progreso.mejor_precision * 100:.1f}%" if progreso.mejor_precision else "0%",
                "tiene_estrella_dorada": (progreso.mejor_precision or 0) >= 0.95,
                "fecha_desbloqueo": progreso.fecha_desbloqueo,
                "fecha_completada": progreso.fecha_completada,
                "puntos_examenes": int(puntos_examenes_leccion),
                "puntos_clases": int(stats_clases.total_puntos_clases or 0)
            })
            
            if completada_actual:
                lecciones_completadas += 1
            total_puntos += progreso.total_puntos or 0
            total_estrellas += progreso.estrellas or 0
        
        db.commit()
        
        total_lecciones = len(lecciones)
        
        porcentaje_general = (clases_completadas_global / total_clases_global * 100) if total_clases_global > 0 else 0
        
        total_examenes_aprobados = db.query(ResultadoExamen).filter(
            ResultadoExamen.usuario_id == usuario_actual.id,
            ResultadoExamen.aprobado == True
        ).count()
        
        total_examenes = db.query(Examen).filter(Examen.activo == True).count()
        
        resultado = {
            "total_lecciones": total_lecciones,
            "lecciones_completadas": lecciones_completadas,
            "porcentaje_completado": round(porcentaje_general, 1),
            "total_puntos": total_puntos + int(puntos_examenes),
            "total_estrellas": total_estrellas,
            "progresos": progresos_data,
            "estadisticas_clases": {
                "clases_completadas": clases_completadas_global,
                "total_clases": total_clases_global,
                "porcentaje_clases": round(porcentaje_general, 1)
            },
            "estadisticas_examenes": {
                "total_examenes": total_examenes,
                "examenes_aprobados": total_examenes_aprobados,
                "puntos_examenes": int(puntos_examenes),
                "porcentaje_examenes_completados": round((total_examenes_aprobados / total_examenes * 100), 1) if total_examenes > 0 else 0
            }
        }
        
        return resultado
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener resumen: {str(e)}"
        )
@router.post("/actualizar-progreso")
def actualizar_progreso_lecciones(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Forzar actualización de todos los progresos de lecciones"""
    try:
        lecciones = db.query(Leccion).filter(Leccion.activa == True).all()
        
        for leccion in lecciones:
            progreso = db.query(ProgresoLeccion).filter(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.leccion_id == leccion.id
            ).first()
            
            if progreso:
                # Recalcular estadísticas actuales
                clases_completadas_count = db.query(ProgresoClase).join(Clase).filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    Clase.leccion_id == leccion.id,
                    ProgresoClase.completada == True,
                    Clase.activa == True
                ).count()
                
                total_clases = db.query(Clase).filter(
                    Clase.leccion_id == leccion.id,
                    Clase.activa == True
                ).count()
                
                # Actualizar campos
                progreso.clases_completadas = clases_completadas_count
                progreso.total_clases = total_clases
                
                # Recalcular si está completada
                progreso.completada = clases_completadas_count >= total_clases and total_clases > 0
                
                if progreso.completada and not progreso.fecha_completada:
                    progreso.fecha_completada = datetime.now()
                elif not progreso.completada:
                    progreso.fecha_completada = None
        
        db.commit()
        
        return {
            "mensaje": "Progreso actualizado exitosamente",
            "detalle": f"Se actualizaron {len(lecciones)} lecciones"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar progreso: {str(e)}"
        )
    
@router.get("/estadisticas/gamificacion", response_model=RespuestaAPI)
async def obtener_estadisticas_gamificacion(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener estadísticas de gamificación del usuario"""
    try:
        # Obtener progreso general del usuario
        lecciones = db.query(Leccion).filter(Leccion.activa == True).all()
        total_lecciones = len(lecciones)
        
        # Calcular lecciones completadas
        lecciones_completadas = db.query(ProgresoLeccion).filter(
            ProgresoLeccion.usuario_id == usuario_actual.id,
            ProgresoLeccion.completada == True
        ).count()
        
        # Calcular clases completadas
        clases_completadas = db.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.completada == True
        ).count()
        
        total_clases = db.query(Clase).filter(Clase.activa == True).count()
        
        # ✅ CORREGIDO: Calcular puntos y XP usando subqueries para consistencia
        # Puntos de clases
        puntos_clases_query = db.query(
            func.sum(ProgresoClase.puntos_ganados)
        ).filter(ProgresoClase.usuario_id == usuario_actual.id).scalar()
        puntos_clases = puntos_clases_query or 0
        
        # XP de clases
        xp_clases_query = db.query(
            func.sum(ProgresoClase.xp_ganado)
        ).filter(ProgresoClase.usuario_id == usuario_actual.id).scalar()
        xp_clases = xp_clases_query or 0
        
        # ✅ CORREGIDO: Puntos de exámenes sin duplicación
        puntos_examenes_query = db.query(
            func.sum(ResultadoExamen.puntuacion_obtenida)
        ).filter(
            ResultadoExamen.usuario_id == usuario_actual.id,
            ResultadoExamen.aprobado == True
        ).scalar()
        puntos_examenes = puntos_examenes_query or 0
        
        # ✅ DECISIÓN IMPORTANTE: Definir qué mostrar como "Puntos Totales"
        # Opción A: Solo puntos de clases (para consistencia con el frontend)
        puntos_totales = int(puntos_clases)
        # Opción B: Puntos de clases + exámenes (para ranking)
        # puntos_totales = int(puntos_clases + puntos_examenes)
        
        # ✅ XP TOTAL debe ser consistente - solo de clases
        xp_total = int(xp_clases)
        
        # Calcular tiempo total de práctica (en horas)
        tiempo_total_segundos = db.query(func.sum(ProgresoClase.tiempo_total_practica)).filter(
            ProgresoClase.usuario_id == usuario_actual.id
        ).scalar() or 0
        tiempo_total_horas = round(tiempo_total_segundos / 3600, 1)
        
        # Calcular precisión global
        precision_global_query = db.query(func.avg(ProgresoClase.mejor_precision)).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.mejor_precision > 0
        ).scalar()
        precision_global = round(float(precision_global_query or 0) * 100, 1)
        
        # Estadísticas de exámenes
        total_examenes = db.query(Examen).filter(Examen.activo == True).count()
        examenes_aprobados = db.query(ResultadoExamen).filter(
            ResultadoExamen.usuario_id == usuario_actual.id,
            ResultadoExamen.aprobado == True
        ).count()
        
        # ✅ CORREGIDO: Cálculo de nivel basado en XP REAL
        nivel_actual = max(1, (xp_total // 100) + 1)  # 100 XP por nivel
        progreso_nivel = xp_total % 100
        
        # Calcular racha
        racha_actual = getattr(usuario_actual, 'racha_actual', 0)
        mejor_racha = getattr(usuario_actual, 'mejor_racha', 0)
        
        # Calcular logros
        logros_desbloqueados = 0
        total_logros = 10
        
        if lecciones_completadas >= 1:
            logros_desbloqueados += 1
        if precision_global >= 80:
            logros_desbloqueados += 1
        if racha_actual >= 7:
            logros_desbloqueados += 1
        if examenes_aprobados >= 1:
            logros_desbloqueados += 1
        
        estadisticas = {
            "puntos_totales": puntos_totales,  # ✅ SOLO puntos de clases para consistencia
            "xp_total": xp_total,  # ✅ XP real solo de clases
            "nivel_actual": nivel_actual,
            "progreso_nivel": progreso_nivel,
            "lecciones_completadas": lecciones_completadas,
            "total_lecciones": total_lecciones,
            "clases_completadas": clases_completadas,
            "total_clases": total_clases,
            "racha_actual": racha_actual,
            "mejor_racha": mejor_racha,
            "tiempo_total_practica_horas": tiempo_total_horas,
            "precision_global": precision_global,
            "logros_desbloqueados": logros_desbloqueados,
            "total_logros": total_logros,
            "rank_global": None,
            # ✅ INFORMACIÓN CLARA Y SEPARADA
            "estadisticas_examenes": {
                "total_examenes": total_examenes,
                "examenes_aprobados": examenes_aprobados,
                "puntos_examenes": int(puntos_examenes),
                "porcentaje_examenes_completados": round((examenes_aprobados / total_examenes * 100), 1) if total_examenes > 0 else 0
            },
            "desglose_puntos": {
                "puntos_clases": int(puntos_clases),
                "puntos_examenes": int(puntos_examenes),
                "total_puntos_clases_examenes": int(puntos_clases + puntos_examenes)  # Para referencia
            },
            "desglose_xp": {
                "xp_clases": int(xp_clases),
                "xp_total": int(xp_total)
            }
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas de gamificación obtenidas",
            datos=estadisticas
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas de gamificación: {str(e)}"
        )
    

@router.get("/ranking", response_model=RespuestaLista)
async def obtener_ranking(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    limite: int = 10
):
    """Obtener ranking de usuarios por puntos (clases + exámenes) - EXCLUYE ADMINISTRADORES"""
    try:
        # ✅ CORREGIDO: Calcular puntos de clases y exámenes por separado para evitar duplicación
        # Primero obtener puntos de clases
        puntos_clases_subquery = db.query(
            ProgresoClase.usuario_id,
            func.sum(ProgresoClase.puntos_ganados).label('puntos_clases')
        ).group_by(ProgresoClase.usuario_id).subquery()
        
        # Luego obtener puntos de exámenes
        puntos_examenes_subquery = db.query(
            ResultadoExamen.usuario_id,
            func.sum(ResultadoExamen.puntuacion_obtenida).label('puntos_examenes')
        ).filter(ResultadoExamen.aprobado == True).group_by(ResultadoExamen.usuario_id).subquery()
        
        # XP de clases
        xp_clases_subquery = db.query(
            ProgresoClase.usuario_id,
            func.sum(ProgresoClase.xp_ganado).label('xp_total')
        ).group_by(ProgresoClase.usuario_id).subquery()
        
        # Query principal con LEFT JOINs para evitar duplicación
        ranking_query = db.query(
            Usuario.id,
            Usuario.nombres,
            Usuario.apellido_paterno,
            Usuario.apellido_materno,
            func.coalesce(puntos_clases_subquery.c.puntos_clases, 0).label('puntos_clases'),
            func.coalesce(puntos_examenes_subquery.c.puntos_examenes, 0).label('puntos_examenes'),
            func.coalesce(xp_clases_subquery.c.xp_total, 0).label('xp_total')
        ).outerjoin(
            puntos_clases_subquery, Usuario.id == puntos_clases_subquery.c.usuario_id
        ).outerjoin(
            puntos_examenes_subquery, Usuario.id == puntos_examenes_subquery.c.usuario_id
        ).outerjoin(
            xp_clases_subquery, Usuario.id == xp_clases_subquery.c.usuario_id
        ).filter(
            and_(
                Usuario.activo == True,
                Usuario.es_admin == False  # ✅ EXCLUIR ADMINISTRADORES
            )
        ).order_by(
            # Ordenar por puntos totales (clases + exámenes)
            (func.coalesce(puntos_clases_subquery.c.puntos_clases, 0) + 
             func.coalesce(puntos_examenes_subquery.c.puntos_examenes, 0)).desc()
        ).limit(limite)
        
        resultados = ranking_query.all()
        
        ranking_data = []
        for posicion, (usuario_id, nombres, apellido_paterno, apellido_materno, puntos_clases, puntos_examenes, xp_total) in enumerate(resultados, 1):
            # ✅ CORREGIDO: Calcular puntos totales sin duplicación
            puntos_totales_ranking = (puntos_clases or 0) + (puntos_examenes or 0)
            
            # ✅ CORREGIDO: XP solo para calcular nivel
            xp_real = xp_total or 0
            nivel = max(1, (xp_real // 100) + 1)  # 100 XP por nivel
            
            # Calcular lecciones completadas
            lecciones_completadas = db.query(ProgresoLeccion).filter(
                ProgresoLeccion.usuario_id == usuario_id,
                ProgresoLeccion.completada == True
            ).count()
            
            usuario_info = {
                "usuario_id": usuario_id,
                "nombre_usuario": f"{nombres} {apellido_paterno}",
                "puntos_totales": int(puntos_totales_ranking),  # ✅ Puntos para ranking
                "nivel": nivel,
                "lecciones_completadas": lecciones_completadas,
                "racha_actual": getattr(usuario_actual, 'racha_actual', 0),
                "posicion": posicion,
                "avatar_url": None,
                # ✅ INFORMACIÓN CLARA
                "desglose_puntos": {
                    "puntos_clases": int(puntos_clases or 0),
                    "puntos_examenes": int(puntos_examenes or 0),
                    "xp_total": int(xp_real)  # XP real para referencia
                }
            }
            ranking_data.append(usuario_info)
        
        # ✅ MODIFICADO: Solo agregar usuario actual si NO es administrador
        usuario_actual_es_admin = getattr(usuario_actual, 'esAdmin', False)
        
        if not usuario_actual_es_admin:
            # Agregar la posición del usuario actual si no está en el top y NO es admin
            usuario_en_ranking = any(user["usuario_id"] == usuario_actual.id for user in ranking_data)
            
            if not usuario_en_ranking:
                # Obtener estadísticas del usuario actual usando el mismo método
                puntos_clases_usuario_query = db.query(
                    func.sum(ProgresoClase.puntos_ganados)
                ).filter(ProgresoClase.usuario_id == usuario_actual.id).scalar()
                puntos_clases_usuario = puntos_clases_usuario_query or 0
                
                puntos_examenes_usuario_query = db.query(
                    func.sum(ResultadoExamen.puntuacion_obtenida)
                ).filter(
                    ResultadoExamen.usuario_id == usuario_actual.id,
                    ResultadoExamen.aprobado == True
                ).scalar()
                puntos_examenes_usuario = puntos_examenes_usuario_query or 0
                
                puntos_totales_usuario = puntos_clases_usuario + puntos_examenes_usuario
                
                xp_usuario_query = db.query(
                    func.sum(ProgresoClase.xp_ganado)
                ).filter(ProgresoClase.usuario_id == usuario_actual.id).scalar()
                xp_usuario = xp_usuario_query or 0
                
                lecciones_completadas_usuario = db.query(ProgresoLeccion).filter(
                    ProgresoLeccion.usuario_id == usuario_actual.id,
                    ProgresoLeccion.completada == True
                ).count()
                
                # Obtener posición global del usuario (excluyendo administradores)
                # Usar el mismo método de cálculo para consistencia
                usuarios_con_mas_puntos = db.query(Usuario.id).select_from(Usuario).outerjoin(
                    puntos_clases_subquery, Usuario.id == puntos_clases_subquery.c.usuario_id
                ).outerjoin(
                    puntos_examenes_subquery, Usuario.id == puntos_examenes_subquery.c.usuario_id
                ).filter(
                    and_(
                        Usuario.activo == True,
                        Usuario.es_admin == False,
                        Usuario.id != usuario_actual.id
                    )
                ).filter(
                    (func.coalesce(puntos_clases_subquery.c.puntos_clases, 0) + 
                     func.coalesce(puntos_examenes_subquery.c.puntos_examenes, 0)) > puntos_totales_usuario
                ).count()
                
                posicion_usuario = usuarios_con_mas_puntos + 1
                
                usuario_actual_info = {
                    "usuario_id": usuario_actual.id,
                    "nombre_usuario": f"{usuario_actual.nombres} {usuario_actual.apellido_paterno}",
                    "puntos_totales": int(puntos_totales_usuario),
                    "nivel": min(100, (xp_usuario // 100) + 1),
                    "lecciones_completadas": lecciones_completadas_usuario,
                    "racha_actual": getattr(usuario_actual, 'racha_actual', 0),
                    "posicion": posicion_usuario,
                    "avatar_url": None,
                    "es_usuario_actual": True,
                    # ✅ NUEVO: Desglose de puntos
                    "desglose_puntos": {
                        "puntos_clases": int(puntos_clases_usuario),
                        "puntos_examenes": int(puntos_examenes_usuario)
                    }
                }
                
                ranking_data.append(usuario_actual_info)
        
        return RespuestaLista(
            exito=True,
            mensaje=f"Ranking de {len(ranking_data)} usuarios (excluyendo administradores)",
            datos=ranking_data,
            total=len(ranking_data),
            pagina=1,
            por_pagina=limite
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener ranking: {str(e)}"
        )
    
@router.get("/usuario/estadisticas-detalladas", response_model=RespuestaAPI)
async def obtener_estadisticas_detalladas_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Obtener estadísticas detalladas del usuario"""
    try:
        # Progreso por lección
        lecciones = db.query(Leccion).filter(Leccion.activa == True).all()
        
        progreso_lecciones = []
        for leccion in lecciones:
            progreso = db.query(ProgresoLeccion).filter(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.leccion_id == leccion.id
            ).first()
            
            clases_completadas = db.query(ProgresoClase).join(Clase).filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                Clase.leccion_id == leccion.id,
                ProgresoClase.completada == True
            ).count()
            
            total_clases = db.query(Clase).filter(
                Clase.leccion_id == leccion.id,
                Clase.activa == True
            ).count()
            
            progreso_lecciones.append({
                "leccion_id": leccion.id,
                "titulo": leccion.titulo,
                "clases_completadas": clases_completadas,
                "total_clases": total_clases,
                "porcentaje_completado": round((clases_completadas / total_clases * 100), 1) if total_clases > 0 else 0,
                "completada": clases_completadas >= total_clases and total_clases > 0
            })
        
        # Estadísticas de práctica
        total_practicas = db.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id
        ).count()
        
        practicas_exitosas = db.query(ProgresoClase).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.intentos_exitosos > 0
        ).count()
        
        tasa_exito = round((practicas_exitosas / total_practicas * 100), 1) if total_practicas > 0 else 0
        
        # Tiempo de práctica por día (últimos 7 días)
        from datetime import timedelta
        fecha_inicio = datetime.now() - timedelta(days=7)
        
        tiempos_por_dia = db.query(
            func.date(ProgresoClase.ultima_practica).label('fecha'),
            func.sum(ProgresoClase.tiempo_total_practica).label('tiempo_total')
        ).filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.ultima_practica >= fecha_inicio
        ).group_by(
            func.date(ProgresoClase.ultima_practica)
        ).order_by(
            func.date(ProgresoClase.ultima_practica)
        ).all()
        
        tiempo_practica_7_dias = [
            {
                "fecha": fecha.strftime('%Y-%m-%d'),
                "tiempo_minutos": round(tiempo_total / 60, 1)
            }
            for fecha, tiempo_total in tiempos_por_dia
        ]
        
        estadisticas_detalladas = {
            "progreso_lecciones": progreso_lecciones,
            "resumen_practicas": {
                "total_practicas": total_practicas,
                "practicas_exitosas": practicas_exitosas,
                "tasa_exito": tasa_exito,
                "precision_promedio": db.query(func.avg(ProgresoClase.mejor_precision)).filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    ProgresoClase.mejor_precision > 0
                ).scalar() or 0
            },
            "tiempo_practica_7_dias": tiempo_practica_7_dias,
            "logros": [
                {
                    "id": 1,
                    "nombre": "Primeros Pasos",
                    "descripcion": "Completa tu primera lección",
                    "desbloqueado": any(leccion["completada"] for leccion in progreso_lecciones),
                    "fecha_desbloqueo": None
                },
                {
                    "id": 2,
                    "nombre": "Precisión Maestra",
                    "descripcion": "Alcanza 90% de precisión en una lección",
                    "desbloqueado": any(leccion.get("mejor_precision", 0) >= 0.9 for leccion in progreso_lecciones),
                    "fecha_desbloqueo": None
                }
            ]
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas detalladas obtenidas",
            datos=estadisticas_detalladas
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas detalladas: {str(e)}"
        )
@router.get("/debug/calculo-puntos")
async def debug_calculo_puntos(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    """Debug endpoint para verificar cálculos de puntos y XP"""
    
    # Obtener todos los progresos de clases
    progresos_clases = db.query(ProgresoClase).filter(
        ProgresoClase.usuario_id == usuario_actual.id
    ).all()
    
    debug_info = {
        "progresos_clases": [],
        "resumen": {
            "total_progresos": len(progresos_clases),
            "puntos_acumulados": 0,
            "xp_acumulado": 0
        }
    }
    
    for progreso in progresos_clases:
        clase_info = {
            "clase_id": progreso.clase_id,
            "completada": progreso.completada,
            "intentos_realizados": progreso.intentos_realizados,
            "puntos_ganados": progreso.puntos_ganados,
            "xp_ganado": progreso.xp_ganado,
            "mejor_precision": progreso.mejor_precision
        }
        debug_info["progresos_clases"].append(clase_info)
        debug_info["resumen"]["puntos_acumulados"] += progreso.puntos_ganados
        debug_info["resumen"]["xp_acumulado"] += progreso.xp_ganado
    
    # Puntos de exámenes
    puntos_examenes = db.query(func.sum(ResultadoExamen.puntuacion_obtenida)).filter(
        ResultadoExamen.usuario_id == usuario_actual.id,
        ResultadoExamen.aprobado == True
    ).scalar() or 0
    
    debug_info["examenes"] = {
        "puntos_examenes": int(puntos_examenes)
    }
    
    debug_info["totales"] = {
        "puntos_totales_clases": debug_info["resumen"]["puntos_acumulados"],
        "xp_total_clases": debug_info["resumen"]["xp_acumulado"],
        "puntos_totales_con_examenes": debug_info["resumen"]["puntos_acumulados"] + puntos_examenes
    }
    
    return debug_info