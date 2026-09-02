from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import time
import base64
import cv2
import numpy as np

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.entrenamiento import ModeloIA
from ..modelos.categoria import Categoria
from ..esquemas.respuesta_schemas import RespuestaAPI
from ..servicios.reconocimiento_adaptativo import ReconocimientoAdaptativoIA

router = APIRouter(prefix="/traductor", tags=["Traductor"])
logger = logging.getLogger(__name__)

_cache_modelos = {
    'modelos': {},
    'ultimo_acceso': {}
}

def obtener_reconocedor_multi(bd: Session, categoria_id: Optional[int] = None) -> Tuple[ReconocimientoAdaptativoIA, int]:
    try:
        modelo_a_usar = None
        
        if categoria_id:
            categoria = bd.query(Categoria).filter(Categoria.id == categoria_id).first()
            if categoria and categoria.modelo_ia_id:
                modelo_a_usar = bd.query(ModeloIA).filter(
                    ModeloIA.id == categoria.modelo_ia_id,
                    ModeloIA.activo == True
                ).first()
        
        if not modelo_a_usar:
            modelo_a_usar = bd.query(ModeloIA).filter(ModeloIA.activo == True).first()
        
        if not modelo_a_usar:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay modelo disponible para reconocimiento"
            )
        
        if modelo_a_usar.id in _cache_modelos['modelos']:
            _cache_modelos['ultimo_acceso'][modelo_a_usar.id] = datetime.now()
            return _cache_modelos['modelos'][modelo_a_usar.id], modelo_a_usar.id
        
        reconocedor = ReconocimientoAdaptativoIA(ruta_modelo=modelo_a_usar.ruta_archivo)
        
        _cache_modelos['modelos'][modelo_a_usar.id] = reconocedor
        _cache_modelos['ultimo_acceso'][modelo_a_usar.id] = datetime.now()
        
        return reconocedor, modelo_a_usar.id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo reconocedor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cargando modelo: {str(e)}"
        )

def predecir_con_multiples_modelos(bd: Session, frames: List, sena_esperada: Optional[str] = None) -> Dict[str, Any]:
    try:
        modelos_activos = bd.query(ModeloIA).filter(ModeloIA.activo == True).all()
        
        if not modelos_activos:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay modelos activos disponibles"
            )
        
        mejor_resultado = None
        todas_predicciones = []
        
        for modelo in modelos_activos:
            try:
                if modelo.id in _cache_modelos['modelos']:
                    reconocedor = _cache_modelos['modelos'][modelo.id]
                else:
                    reconocedor = ReconocimientoAdaptativoIA(ruta_modelo=modelo.ruta_archivo)
                    _cache_modelos['modelos'][modelo.id] = reconocedor
                    _cache_modelos['ultimo_acceso'][modelo.id] = datetime.now()
                
                sena_detectada, confianza, detalles = reconocedor.predecir_desde_secuencia(frames, sena_esperada=sena_esperada)
                
                resultado_modelo = {
                    'modelo_id': modelo.id,
                    'modelo_nombre': modelo.nombre,
                    'sena_detectada': sena_detectada,
                    'confianza': confianza,
                    'detalles': detalles
                }
                
                todas_predicciones.append(resultado_modelo)
                
                if mejor_resultado is None or confianza > mejor_resultado['confianza']:
                    mejor_resultado = resultado_modelo
                
            except Exception as e:
                continue
        
        if not mejor_resultado:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ningún modelo pudo procesar la solicitud"
            )
        
        todas_predicciones.sort(key=lambda x: x['confianza'], reverse=True)
        
        return {
            'mejor_prediccion': mejor_resultado,
            'todas_predicciones': todas_predicciones[:5],
            'total_modelos_consultados': len(todas_predicciones)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción multi-modelo: {str(e)}")
        raise

@router.post("/senas-a-texto", response_model=RespuestaAPI)
async def traducir_senas_a_texto(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    start_time = time.time()
    
    try:
        video_base64 = data.get("video_base64")
        frames_base64 = data.get("frames_base64", [])
        imagen_base64 = data.get("imagen_base64")
        sena_esperada = data.get("sena_esperada")
        categoria_id = data.get("categoria_id")
        usar_multiples_modelos = data.get("usar_multiples_modelos", True)  # POR DEFECTO TRUE
        configuracion = data.get("configuracion", {})
        
        if not video_base64 and not frames_base64 and not imagen_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requiere video, frames o imagen para reconocimiento"
            )
        
        frames = []
        if frames_base64:
            frames_a_procesar = frames_base64[-16:]
            for frame_b64 in frames_a_procesar:
                try:
                    if ',' in frame_b64:
                        frame_b64 = frame_b64.split(',')[1]
                    frame_bytes = base64.b64decode(frame_b64)
                    frame_np = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                    if frame is not None:
                        if frame.shape[0] != 224 or frame.shape[1] != 224:
                            frame = cv2.resize(frame, (224, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                except Exception as e:
                    continue
        
        if len(frames) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Frames insuficientes ({len(frames)}/8)"
            )
        
        # ✅ CORRECCIÓN: SIEMPRE usar múltiples modelos a menos que se especifique categoría
        if not categoria_id:
            # Modo traductor libre: usar todos los modelos
            resultado_multi = predecir_con_multiples_modelos(bd, frames, sena_esperada)
            
            mejor = resultado_multi['mejor_prediccion']
            sena_detectada = mejor['sena_detectada']
            confianza = mejor['confianza']
            detalles = mejor['detalles']
            modelo_usado = mejor['modelo_nombre']
            modelo_id_usado = mejor['modelo_id']
            
            logger.info(f"Usando múltiples modelos. Mejor: {modelo_usado} - {sena_detectada} ({confianza*100:.1f}%)")
            
        else:
            # Modo específico: usar solo el modelo de la categoría
            reconocedor, modelo_id_usado = obtener_reconocedor_multi(bd, categoria_id)
            
            sena_detectada, confianza, detalles = reconocedor.predecir_desde_secuencia(frames, sena_esperada=sena_esperada)
            
            modelo_db = bd.query(ModeloIA).filter(ModeloIA.id == modelo_id_usado).first()
            modelo_usado = modelo_db.nombre if modelo_db else "Desconocido"
            
            logger.info(f"Usando categoría específica: {modelo_usado} - {sena_detectada} ({confianza*100:.1f}%)")
        
        processing_time = time.time() - start_time
        
        confianza_minima = configuracion.get("confianza_minima", 0.25)
        
        if confianza >= 0.80:
            calidad = "excelente"
            mensaje = "Alta confianza"
        elif confianza >= 0.60:
            calidad = "buena"
            mensaje = "Buena confianza"
        elif confianza >= 0.40:
            calidad = "moderada"
            mensaje = "Confianza moderada"
        elif confianza >= confianza_minima:
            calidad = "baja"
            mensaje = "Baja confianza"
        else:
            calidad = "muy_baja"
            mensaje = "Seña no reconocida"
        
        texto_traducido = sena_detectada
        if confianza < confianza_minima:
            texto_traducido = f"{sena_detectada}?"
        
        resultado = {
            "modo": "senas-a-texto",
            "texto_traducido": texto_traducido,
            "sena_detectada": sena_detectada,
            "confianza": round(confianza, 4),
            "confianza_raw": round(confianza, 4),
            "porcentaje": round(confianza * 100, 2),
            "calidad": calidad,
            "mensaje": mensaje,
            "modelo_usado": modelo_usado,
            "modelo_id": modelo_id_usado,
            "categoria_id": categoria_id,
            "alternativas": detalles.get("alternativas", [])[:3],
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        # ✅ SIEMPRE incluir información de múltiples modelos cuando no hay categoría
        if not categoria_id and 'resultado_multi' in locals():
            resultado['modelos_consultados'] = resultado_multi['total_modelos_consultados']
            resultado['todas_predicciones'] = [
                {
                    'modelo': pred['modelo_nombre'],
                    'sena': pred['sena_detectada'],
                    'confianza': round(pred['confianza'], 4),
                    'porcentaje': round(pred['confianza'] * 100, 2)
                }
                for pred in resultado_multi['todas_predicciones'][:3]
            ]
        
        return RespuestaAPI(
            exito=True,
            mensaje="Reconocimiento completado",
            datos=resultado
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en traducción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en reconocimiento: {str(e)}"
        )

@router.get("/modelos-activos")
async def obtener_modelos_activos(
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        modelos = bd.query(ModeloIA).filter(ModeloIA.activo == True).all()
        
        modelos_info = []
        for modelo in modelos:
            categorias = bd.query(Categoria).filter(
                Categoria.modelo_ia_id == modelo.id,
                Categoria.activa == True
            ).all()
            
            # Obtener información del reconocedor cargado
            reconocedor_info = {}
            if modelo.id in _cache_modelos['modelos']:
                reconocedor = _cache_modelos['modelos'][modelo.id]
                reconocedor_info = {
                    'clases_cargadas': len(reconocedor.clases),
                    'arquitectura': reconocedor.arquitectura,
                    'accuracy': round(reconocedor.accuracy * 100, 2) if hasattr(reconocedor, 'accuracy') else 0
                }
            
            modelos_info.append({
                "id": modelo.id,
                "nombre": modelo.nombre,
                "num_clases": modelo.num_clases,
                "accuracy": round(modelo.accuracy * 100, 2) if modelo.accuracy else 0,
                "categorias_asignadas": [{"id": cat.id, "nombre": cat.nombre} for cat in categorias],
                "reconocedor_cargado": modelo.id in _cache_modelos['modelos'],
                "info_reconocedor": reconocedor_info
            })
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Se encontraron {len(modelos_info)} modelos activos",
            datos={
                "total_modelos": len(modelos_info),
                "modelos": modelos_info
            }
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos: {e}")
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error: {str(e)}"
        )

@router.post("/probar-todos-modelos")
async def probar_todos_modelos(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        frames_base64 = data.get("frames_base64", [])
        
        if not frames_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requieren frames para la prueba"
            )
        
        frames = []
        for frame_b64 in frames_base64[-12:]:
            try:
                if ',' in frame_b64:
                    frame_b64 = frame_b64.split(',')[1]
                frame_bytes = base64.b64decode(frame_b64)
                frame_np = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                if frame is not None:
                    if frame.shape[0] != 224 or frame.shape[1] != 224:
                        frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            except Exception as e:
                continue
        
        if len(frames) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Frames insuficientes para prueba"
            )
        
        resultado_multi = predecir_con_multiples_modelos(bd, frames)
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Prueba completada con {resultado_multi['total_modelos_consultados']} modelos",
            datos=resultado_multi
        )
        
    except Exception as e:
        logger.error(f"Error en prueba de modelos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/limpiar-cache")
async def limpiar_cache_traductor(modelo_id: Optional[int] = None):
    try:
        if modelo_id:
            if modelo_id in _cache_modelos['modelos']:
                del _cache_modelos['modelos'][modelo_id]
                if modelo_id in _cache_modelos['ultimo_acceso']:
                    del _cache_modelos['ultimo_acceso'][modelo_id]
                mensaje = f"Cache del modelo {modelo_id} limpiado"
            else:
                mensaje = f"Modelo {modelo_id} no estaba en cache"
        else:
            _cache_modelos['modelos'].clear()
            _cache_modelos['ultimo_acceso'].clear()
            mensaje = "Cache de todos los modelos limpiado"
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        return RespuestaAPI(
            exito=True,
            mensaje=mensaje
        )
        
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error: {str(e)}"
        )