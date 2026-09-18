from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import time

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..esquemas.respuesta_schemas import RespuestaAPI
from ..servicios.gestor_reconocimiento import (
    obtener_reconocedor,
    determinar_tipo_sena,
    procesar_frame_base64,
    extraer_keypoints_frame,
    limpiar_cache,
    info_cache,
)

router = APIRouter(prefix="/traductor", tags=["Traductor"])
logger = logging.getLogger(__name__)

MIN_FRAMES_CON_MANOS = 4


@router.post("/senas-a-texto", response_model=RespuestaAPI)
async def traducir_senas_a_texto(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    start_time = time.time()
    try:
        frames_base64 = data.get("frames_base64", [])
        imagen_base64 = data.get("imagen_base64")
        sena_esperada = data.get("sena_esperada")
        categoria_id = data.get("categoria_id")
        configuracion = data.get("configuracion", {})
        confianza_minima = configuracion.get("confianza_minima", 0.25)

        if not frames_base64 and not imagen_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requieren frames o imagen"
            )

        frames = []
        if frames_base64:
            for frame_b64 in frames_base64[-16:]:
                frame = procesar_frame_base64(frame_b64)
                if frame is not None:
                    frames.append(frame)
        if not frames and imagen_base64:
            frame = procesar_frame_base64(imagen_base64)
            if frame is not None:
                frames.append(frame)

        if len(frames) < MIN_FRAMES_CON_MANOS:
            return RespuestaAPI(
                exito=False,
                mensaje=f"Frames insuficientes ({len(frames)}/{MIN_FRAMES_CON_MANOS})",
                datos={"confianza": 0.0, "sena_detectada": ""}
            )

        keypoints_secuencia = []
        for frame in frames:
            kp = extraer_keypoints_frame(frame)
            if kp is not None:
                keypoints_secuencia.append(kp)

        if len(keypoints_secuencia) < MIN_FRAMES_CON_MANOS:
            processing_time = time.time() - start_time
            return RespuestaAPI(
                exito=False,
                mensaje=f"No se detectaron manos suficientes ({len(keypoints_secuencia)}/{MIN_FRAMES_CON_MANOS})",
                datos={
                    "confianza": 0.0,
                    "sena_detectada": "",
                    "num_frames_procesados": len(keypoints_secuencia),
                    "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
                    "categoria_id": categoria_id,
                }
            )

        reconocedor, modelo_id = obtener_reconocedor(bd, categoria_id)
        tipo_sena = determinar_tipo_sena(bd, sena_esperada, categoria_id, reconocedor)

        if tipo_sena == 'ESTATICA' and len(keypoints_secuencia) >= 5:
            centro = len(keypoints_secuencia) // 2
            inicio = max(0, centro - 2)
            fin = min(len(keypoints_secuencia), centro + 2)
            keypoints_a_usar = keypoints_secuencia[inicio:fin]
        else:
            keypoints_a_usar = keypoints_secuencia

        sena_detectada, confianza, detalles = reconocedor.predecir_desde_keypoints(
            keypoints_a_usar, tipo_sena
        )
        modo = tipo_sena

        confianza = max(0.0, min(1.0, confianza))
        processing_time = time.time() - start_time

        if confianza >= 0.80:
            calidad, mensaje = "excelente", f"Alta confianza: {sena_detectada}"
        elif confianza >= 0.70:
            calidad, mensaje = "buena", f"Buena confianza: {sena_detectada}"
        elif confianza >= 0.60:
            calidad, mensaje = "moderada", f"Confianza moderada: {sena_detectada}"
        elif confianza >= confianza_minima:
            calidad, mensaje = "baja", f"Baja confianza: {sena_detectada}"
        else:
            calidad, mensaje = "muy_baja", "Seña no reconocida claramente"

        texto_traducido = sena_detectada if confianza >= confianza_minima else f"{sena_detectada}?"

        resultado = {
            "sena_detectada": sena_detectada,
            "texto_traducido": texto_traducido,
            "confianza": round(confianza, 4),
            "porcentaje": round(confianza * 100, 2),
            "calidad": calidad,
            "mensaje": mensaje,
            "modo": modo,
            "num_frames_procesados": len(keypoints_secuencia),
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "modelo_usado": modelo_id,
            "categoria_id": categoria_id,
            "alternativas": detalles.get("alternativas", [])[:3],
            "timestamp": datetime.now().isoformat()
        }

        return RespuestaAPI(exito=True, mensaje="Reconocimiento completado", datos=resultado)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en traducción: {str(e)}", exc_info=True)
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error interno: {str(e)}",
            datos={"confianza": 0.0, "sena_detectada": ""}
        )


@router.get("/modelos-activos")
async def obtener_modelos_activos(
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    from ..modelos.entrenamiento import ModeloIA
    from ..modelos.categoria import Categoria
    from ..servicios.gestor_reconocimiento import _cache

    try:
        modelos = bd.query(ModeloIA).filter(ModeloIA.activo == True).all()

        modelos_info = []
        for modelo in modelos:
            categorias = bd.query(Categoria).filter(
                Categoria.modelo_ia_id == modelo.id,
                Categoria.activa == True
            ).all()

            reconocedor_info = {}
            cargado = modelo.id in _cache['reconocedores']
            if cargado:
                reconocedor = _cache['reconocedores'][modelo.id]
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
                "reconocedor_cargado": cargado,
                "info_reconocedor": reconocedor_info
            })

        return RespuestaAPI(
            exito=True,
            mensaje=f"Se encontraron {len(modelos_info)} modelos activos",
            datos={"total_modelos": len(modelos_info), "modelos": modelos_info}
        )

    except Exception as e:
        logger.error(f"Error obteniendo modelos: {e}")
        return RespuestaAPI(exito=False, mensaje=f"Error: {str(e)}")


@router.post("/probar-todos-modelos")
async def probar_todos_modelos(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint pendiente: falta la función predecir_con_multiples_modelos"
    )


@router.post("/limpiar-cache")
async def limpiar_cache_traductor(modelo_id: Optional[int] = None):
    try:
        mensaje = limpiar_cache(modelo_id)
        return RespuestaAPI(exito=True, mensaje=mensaje)
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return RespuestaAPI(exito=False, mensaje=f"Error: {str(e)}")