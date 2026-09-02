from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import base64
from datetime import datetime
import logging
from pydantic import BaseModel
import time
import mediapipe as mp

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..modelos.entrenamiento import ModeloIA
from ..modelos.categoria import Categoria

from ..servicios.reconocimiento_adaptativo import ReconocimientoAdaptativoIA
from ..servicios.config_tipo_senas import detectar_tipo_sena

router = APIRouter(prefix="/reconocimiento-video", tags=["Reconocimiento Video"])

logger = logging.getLogger(__name__)

_reconocedor_video_cache = {
    'reconocedor': None,
    'modelo_id': None,
    'categoria_id': None,
    'ultimo_uso': None,
    'warmup_realizado': False,
    'cache_key': None
}

_mediapipe_hands = None

def obtener_mediapipe_hands():
    global _mediapipe_hands
    if _mediapipe_hands is None:
        _mediapipe_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _mediapipe_hands

class FrameData(BaseModel):
    frame_base64: str
    timestamp: int
    landmarks: Optional[List[Any]] = None

class SolicitudVideoSecuencia(BaseModel):
    frames: List[FrameData]
    configuracion: Dict[str, Any] = {}
    sena_esperada: Optional[str] = None
    categoria_id: Optional[int] = None

def precalentar_modelo(reconocedor: ReconocimientoAdaptativoIA):
    global _reconocedor_video_cache
    
    if _reconocedor_video_cache.get('warmup_realizado'):
        return
    
    try:
        import torch
        logger.info("Precalentando modelo...")
        
        dummy_keypoints = np.random.randn(6, 126).astype(np.float32)
        dummy_tensor = torch.from_numpy(dummy_keypoints).float()
        dummy_tensor = dummy_tensor.unsqueeze(0).to(reconocedor.device)
        
        with torch.no_grad():
            for _ in range(3):
                _ = reconocedor.predecir(dummy_tensor, 'DINAMICA', None)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _reconocedor_video_cache['warmup_realizado'] = True
        logger.info("Modelo precalentado exitosamente")
        
    except Exception as e:
        logger.warning(f"Error en precalentamiento: {e}")

def obtener_modelo_por_categoria(db: Session, categoria_id: Optional[int] = None) -> ModeloIA:
    if categoria_id:
        categoria = db.query(Categoria).filter(Categoria.id == categoria_id).first()
        
        if categoria and categoria.modelo_ia_id:
            modelo_categoria = db.query(ModeloIA).filter(
                ModeloIA.id == categoria.modelo_ia_id,
                ModeloIA.activo == True
            ).first()
            
            if modelo_categoria:
                logger.info(f"Modelo categoría: {categoria.nombre} -> {modelo_categoria.nombre}")
                return modelo_categoria
    
    modelo_global = db.query(ModeloIA).filter(ModeloIA.activo == True).first()
    if modelo_global:
        logger.info(f"Modelo global: {modelo_global.nombre}")
        return modelo_global
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No hay modelo disponible"
    )

def obtener_reconocedor_video(db: Session, categoria_id: Optional[int] = None) -> ReconocimientoAdaptativoIA:
    try:
        modelo_a_usar = obtener_modelo_por_categoria(db, categoria_id)
        cache_key = f"modelo_{modelo_a_usar.id}_cat_{categoria_id}"
        
        if (_reconocedor_video_cache.get('reconocedor') and 
            _reconocedor_video_cache.get('cache_key') == cache_key):
            _reconocedor_video_cache['ultimo_uso'] = datetime.now()
            return _reconocedor_video_cache['reconocedor']
        
        logger.info(f"Cargando modelo: {modelo_a_usar.nombre}")
        reconocedor = ReconocimientoAdaptativoIA(ruta_modelo=modelo_a_usar.ruta_archivo)
        
        precalentar_modelo(reconocedor)
        
        _reconocedor_video_cache.update({
            'reconocedor': reconocedor,
            'modelo_id': modelo_a_usar.id,
            'categoria_id': categoria_id,
            'cache_key': cache_key,
            'ultimo_uso': datetime.now(),
            'warmup_realizado': True
        })
        
        return reconocedor
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cargando reconocedor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cargando modelo: {str(e)}"
        )

def extraer_keypoints_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    try:
        hands = obtener_mediapipe_hands()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        keypoints = []
        
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        if len(keypoints) == 63:
            keypoints.extend([0.0] * 63)
        elif len(keypoints) != 126:
            return None
        
        return np.array(keypoints, dtype=np.float32)
        
    except Exception as e:
        logger.warning(f"Error extrayendo keypoints: {e}")
        return None

def procesar_frame_base64_optimizado(frame_base64: str) -> Optional[np.ndarray]:
    try:
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
            
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            
        return frame
        
    except Exception as e:
        logger.warning(f"Error procesando frame: {e}")
        return None

@router.post("/secuencia", response_model=Dict[str, Any])
async def reconocer_secuencia_video(
    solicitud: SolicitudVideoSecuencia,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    start_time = time.time()
    
    try:
        num_frames_recibidos = len(solicitud.frames)
        reconocedor = obtener_reconocedor_video(db, solicitud.categoria_id)
        
        tipo_sena = None
        
        if solicitud.sena_esperada:
            tipo_sena = detectar_tipo_sena(solicitud.sena_esperada)
            logger.info(f"Tipo detectado por sena_esperada '{solicitud.sena_esperada}': {tipo_sena}")
        
        if tipo_sena is None:
            clases_estaticas = [clase for clase, tipo in reconocedor.tipos_senas.items() 
                              if tipo == 'ESTATICA']
            total_clases = len(reconocedor.clases)
            
            if total_clases > 0 and len(clases_estaticas) == total_clases:
                tipo_sena = 'ESTATICA'
                logger.info(f"Todas las {total_clases} clases son estáticas, usando ESTATICA")
            else:
                tipo_sena = 'DINAMICA'
                logger.info(f"Usando DINAMICA por defecto (hay {len(clases_estaticas)}/{total_clases} estáticas)")
        
        if solicitud.categoria_id:
            categoria = db.query(Categoria).filter(Categoria.id == solicitud.categoria_id).first()
            if categoria:
                nombre_cat = categoria.nombre.upper()
                categorias_estaticas = ['LETRAS', 'ABECEDARIO', 'ALFABETO', 'NÚMEROS', 'NUMEROS', 'DIGITOS']
                
                if any(cat_est in nombre_cat for cat_est in categorias_estaticas):
                    tipo_sena = 'ESTATICA'
                    logger.info(f"Categoría '{categoria.nombre}' detectada como ESTATICA")
        
        logger.info(f"Modo final seleccionado: {tipo_sena}")
        
        max_frames = 8 if tipo_sena == 'ESTATICA' else 15
        
        if num_frames_recibidos > max_frames:
            frames_a_procesar = solicitud.frames[-max_frames:]
            logger.info(f"Optimizando: {num_frames_recibidos} -> {max_frames} frames")
        else:
            frames_a_procesar = solicitud.frames
        
        keypoints_secuencia = []
        
        for i, frame_data in enumerate(frames_a_procesar):
            frame = procesar_frame_base64_optimizado(frame_data.frame_base64)
            if frame is not None:
                keypoints = extraer_keypoints_frame(frame)
                if keypoints is not None:
                    keypoints_secuencia.append(keypoints)
                else:
                    logger.debug(f"Frame {i}: no se pudieron extraer keypoints")
        
        frames_exitosos = len(keypoints_secuencia)
        
        if frames_exitosos < 4:
            return {
                "exito": False,
                "mensaje": f"Frames insuficientes ({frames_exitosos}/4)",
                "datos": {
                    "confianza": 0.0,
                    "sena_detectada": "",
                    "num_frames_procesados": frames_exitosos,
                    "categoria_id": solicitud.categoria_id
                }
            }
        
        logger.info(f"Frames procesados: {frames_exitosos}, usando modo: {tipo_sena}")
        
        if tipo_sena == 'ESTATICA':
            if frames_exitosos >= 5:
                centro = len(keypoints_secuencia) // 2
                inicio = max(0, centro - 2)
                fin = min(len(keypoints_secuencia), centro + 2)
                keypoints_a_usar = keypoints_secuencia[inicio:fin]
                logger.info(f"Modo ESTATICO: usando {len(keypoints_a_usar)} frames del centro")
            else:
                keypoints_a_usar = keypoints_secuencia
            
            sena_detectada, confianza, detalles = reconocedor.predecir_desde_keypoints(
                keypoints_a_usar, 
                tipo_sena
            )
        else:
            sena_detectada, confianza, detalles = reconocedor.predecir_desde_keypoints(
                keypoints_secuencia, 
                tipo_sena
            )
        
        confianza_validada = max(0.0, min(1.0, confianza))
        processing_time = time.time() - start_time
        
        if confianza_validada >= 0.80:
            calidad = "excelente"
            mensaje = f"Alta confianza: {sena_detectada}"
        elif confianza_validada >= 0.70:
            calidad = "buena" 
            mensaje = f"Buena confianza: {sena_detectada}"
        elif confianza_validada >= 0.60:
            calidad = "moderada"
            mensaje = f"Confianza moderada: {sena_detectada}"
        elif confianza_validada >= 0.50:
            calidad = "baja"
            mensaje = f"Baja confianza: {sena_detectada}"
        else:
            calidad = "muy_baja"
            mensaje = f"Seña no reconocida claramente"
        
        alternativas_filtradas = []
        if "alternativas" in detalles:
            for alt in detalles["alternativas"]:
                if alt.get("confianza", 0) >= 0.2:
                    alternativas_filtradas.append(alt)
        
        alternativas_filtradas = sorted(alternativas_filtradas, 
                                      key=lambda x: x.get("confianza", 0), 
                                      reverse=True)[:3]
        
        modelo_info = {
            "modelo_id": _reconocedor_video_cache.get('modelo_id'),
            "categoria_id": solicitud.categoria_id,
            "modelo_nombre": reconocedor.arquitectura if reconocedor else "desconocido",
            "modo_usado": tipo_sena
        }
        
        resultado = {
            "sena_detectada": sena_detectada,
            "texto_traducido": sena_detectada,
            "confianza": round(confianza_validada, 4),
            "confianza_raw": round(confianza_validada, 4),
            "porcentaje": round(confianza_validada * 100, 2),
            "calidad": calidad,
            "mensaje": mensaje,
            "modo": tipo_sena,
            "num_frames_procesados": frames_exitosos,
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "fps": round(frames_exitosos / processing_time, 2) if processing_time > 0 else 0,
            "alternativas": alternativas_filtradas,
            "modelo_usado": modelo_info,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Reconocimiento [Cat:{solicitud.categoria_id}]: {sena_detectada} ({confianza_validada*100:.1f}%) - Modo: {tipo_sena}")
        
        return {
            "exito": True,
            "mensaje": "Reconocimiento completado",
            "datos": resultado
        }
        
    except Exception as e:
        logger.error(f"Error en reconocimiento [Cat:{solicitud.categoria_id}]: {str(e)}", exc_info=True)
        return {
            "exito": False,
            "mensaje": "Error en el reconocimiento",
            "datos": {
                "confianza": 0.0,
                "sena_detectada": "",
                "num_frames_procesados": 0,
                "categoria_id": solicitud.categoria_id
            },
            "errores": [f"Error interno: {str(e)}"]
        }

@router.get("/info-modelo")
async def obtener_info_modelo(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        reconocedor = obtener_reconocedor_video(db)
        
        modelo_db = db.query(ModeloIA).filter(
            ModeloIA.id == _reconocedor_video_cache['modelo_id']
        ).first()
        
        info = {
            "exito": True,
            "modelo": {
                "id": modelo_db.id if modelo_db else None,
                "nombre": modelo_db.nombre if modelo_db else "desconocido",
                "arquitectura": reconocedor.arquitectura,
                "num_clases": reconocedor.num_clases,
                "clases": reconocedor.clases,
                "num_frames": reconocedor.num_frames,
                "accuracy_entrenamiento": round(reconocedor.accuracy * 100, 2),
                "fecha_entrenamiento": modelo_db.fecha_entrenamiento.isoformat() if modelo_db and modelo_db.fecha_entrenamiento else None,
                "tipos_senas": reconocedor.tipos_senas
            },
            "sistema": {
                "device": str(reconocedor.device),
                "cuda_disponible": str(reconocedor.device).startswith('cuda'),
                "amp_enabled": reconocedor.enable_amp,
                "modo": "keypoints_lstm"
            }
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error obteniendo info: {e}")
        return {
            "exito": False,
            "mensaje": f"Error: {str(e)}"
        }

@router.get("/estado")
async def obtener_estado_reconocimiento(
    db: Session = Depends(obtener_bd)
):
    try:
        reconocedor = _reconocedor_video_cache.get('reconocedor')
        
        if reconocedor is None:
            try:
                reconocedor = obtener_reconocedor_video(db)
            except Exception as e:
                logger.warning(f"No se pudo cargar reconocedor: {e}")
        
        estado = {
            "estado": "activo" if reconocedor is not None else "inactivo",
            "modelo_cargado": reconocedor is not None,
            "arquitectura": reconocedor.arquitectura if reconocedor else "ninguna",
            "clases_cargadas": len(reconocedor.clases) if reconocedor else 0,
            "clases": reconocedor.clases if reconocedor else [],
            "num_frames": reconocedor.num_frames if reconocedor else 0,
            "device": str(reconocedor.device) if reconocedor else "ninguno",
            "ultimo_uso": _reconocedor_video_cache['ultimo_uso'].isoformat() if _reconocedor_video_cache['ultimo_uso'] else None,
            "modo": "keypoints_lstm",
            "timestamp": datetime.now().isoformat()
        }
        
        return estado
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return {
            "estado": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/limpiar-cache")
async def limpiar_cache_modelo():
    try:
        global _mediapipe_hands
        
        _reconocedor_video_cache['reconocedor'] = None
        _reconocedor_video_cache['modelo_id'] = None
        _reconocedor_video_cache['categoria_id'] = None
        _reconocedor_video_cache['ultimo_uso'] = None
        _reconocedor_video_cache['warmup_realizado'] = False
        _reconocedor_video_cache['cache_key'] = None
        
        if _mediapipe_hands:
            _mediapipe_hands.close()
            _mediapipe_hands = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Cache del modelo limpiado")
        
        return {
            "exito": True,
            "mensaje": "Cache limpiado exitosamente"
        }
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return {
            "exito": False,
            "mensaje": f"Error: {str(e)}"
        }