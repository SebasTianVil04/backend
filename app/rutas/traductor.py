import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..modelos.usuario import Usuario
from ..esquemas.respuesta_schemas import RespuestaAPI
from ..servicios.gestor_reconocimiento import (
    obtener_reconocedor,
    determinar_tipo_sena,
    procesar_frame_base64,
    limpiar_cache,
    info_cache,
)
from ..servicios.extractor_secuencias import normalizar_fila

router = APIRouter(prefix="/traductor", tags=["Traductor"])
logger = logging.getLogger(__name__)

MIN_FRAMES_CON_MANOS = 4
MAX_FRAMES_ENTRADA = 120

FEATURES_POR_MANO = 63
FEATURES_TOTALES = 126


def _landmarks_a_fila(landmarks: Any) -> Optional[np.ndarray]:
    if not landmarks or not isinstance(landmarks, (list, tuple)):
        return None

    fila = np.zeros(FEATURES_TOTALES, dtype=np.float32)
    hay_datos = False

    for idx, mano in enumerate(landmarks[:2]):
        if not isinstance(mano, (list, tuple)) or len(mano) < 21:
            continue

        base = idx * FEATURES_POR_MANO
        for j in range(21):
            punto = mano[j]
            if isinstance(punto, dict):
                x = float(punto.get('x', 0.0) or 0.0)
                y = float(punto.get('y', 0.0) or 0.0)
                z = float(punto.get('z', 0.0) or 0.0)
            else:
                x = float(getattr(punto, 'x', 0.0) or 0.0)
                y = float(getattr(punto, 'y', 0.0) or 0.0)
                z = float(getattr(punto, 'z', 0.0) or 0.0)
            fila[base + j * 3] = x
            fila[base + j * 3 + 1] = y
            fila[base + j * 3 + 2] = z
        hay_datos = True

    if not hay_datos:
        return None

    try:
        return normalizar_fila(fila)
    except Exception:
        return fila


def _evaluar_calidad_confianza(confianza: float, confianza_minima: float, sena: str):
    if confianza >= 0.80:
        return "excelente", f"Alta confianza: {sena}"
    if confianza >= 0.70:
        return "buena", f"Buena confianza: {sena}"
    if confianza >= 0.60:
        return "moderada", f"Confianza moderada: {sena}"
    if confianza >= confianza_minima:
        return "baja", f"Baja confianza: {sena}"
    return "muy_baja", "Seña no reconocida claramente"


@router.post("/senas-a-texto", response_model=RespuestaAPI)
async def traducir_senas_a_texto(
    data: Dict[str, Any],
    bd: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    start_time = time.time()
    try:
        frames_base64 = data.get("frames_base64", []) or []
        frames_landmarks = data.get("frames_landmarks", []) or []
        imagen_base64 = data.get("imagen_base64")
        imagen_landmarks = data.get("imagen_landmarks")
        sena_esperada = data.get("sena_esperada")
        configuracion = data.get("configuracion", {}) or {}
        categoria_id = data.get("categoria_id", configuracion.get("categoria_id"))
        confianza_minima = float(configuracion.get("confianza_minima", 0.25))

        if not frames_base64 and not imagen_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requieren frames o imagen"
            )

        keypoints_cliente: List[np.ndarray] = []
        frames_validos: List[np.ndarray] = []

        if frames_base64:
            for idx, frame_b64 in enumerate(frames_base64[-MAX_FRAMES_ENTRADA:]):
                frame = procesar_frame_base64(frame_b64)
                if frame is not None:
                    frames_validos.append(frame)

                if idx < len(frames_landmarks):
                    fila = _landmarks_a_fila(frames_landmarks[idx])
                    if fila is not None:
                        keypoints_cliente.append(fila)

        if not frames_validos and imagen_base64:
            frame = procesar_frame_base64(imagen_base64)
            if frame is not None:
                frames_validos.append(frame)
            if imagen_landmarks:
                fila = _landmarks_a_fila(imagen_landmarks)
                if fila is not None:
                    keypoints_cliente.append(fila)

        reconocedor, modelo_id = obtener_reconocedor(bd, categoria_id)

        usar_landmarks = len(keypoints_cliente) >= MIN_FRAMES_CON_MANOS

        if usar_landmarks:
            resultado = reconocedor._resultado_desde_keypoints(keypoints_cliente)
        else:
            if len(frames_validos) < MIN_FRAMES_CON_MANOS:
                return RespuestaAPI(
                    exito=False,
                    mensaje=f"Frames insuficientes ({len(frames_validos)}/{MIN_FRAMES_CON_MANOS})",
                    datos={"confianza": 0.0, "sena_detectada": ""}
                )
            resultado = reconocedor._resultado_desde_frames(frames_validos)

        if resultado.frames_detectados < MIN_FRAMES_CON_MANOS:
            processing_time = time.time() - start_time
            return RespuestaAPI(
                exito=False,
                mensaje=f"No se detectaron manos suficientes ({resultado.frames_detectados}/{MIN_FRAMES_CON_MANOS})",
                datos={
                    "confianza": 0.0,
                    "sena_detectada": "",
                    "num_frames_procesados": resultado.frames_detectados,
                    "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
                    "categoria_id": categoria_id,
                }
            )

        keypoints_secuencia = [
            fila for fila, estado in zip(resultado.secuencia, resultado.estado)
            if estado > 0
        ]

        tipo_sena_inicial = determinar_tipo_sena(
            bd, sena_esperada, categoria_id, reconocedor,
            keypoints_secuencia=keypoints_secuencia
        )

        tensor = reconocedor._tensor_desde_resultado(resultado)
        sena_detectada, confianza, detalles = reconocedor.predecir(
            tensor, tipo_sena_inicial, sena_esperada
        )

        tipo_sena = detalles.get("tipo_sena", tipo_sena_inicial)

        if not sena_detectada or sena_detectada == "desconocido":
            processing_time = time.time() - start_time
            return RespuestaAPI(
                exito=False,
                mensaje=detalles.get("error") or "No se pudo reconocer la seña",
                datos={
                    "confianza": 0.0,
                    "sena_detectada": "",
                    "num_frames_procesados": resultado.frames_detectados,
                    "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
                    "categoria_id": categoria_id,
                }
            )

        confianza = max(0.0, min(1.0, confianza))
        processing_time = time.time() - start_time

        calidad, mensaje = _evaluar_calidad_confianza(confianza, confianza_minima, sena_detectada)

        texto_traducido = sena_detectada if confianza >= confianza_minima else f"{sena_detectada}?"

        alternativas = sorted(
            [a for a in detalles.get("alternativas", []) if a.get("confianza", 0) >= 0.15],
            key=lambda x: x.get("confianza", 0),
            reverse=True
        )[:3]

        resultado_final = {
            "sena_detectada": sena_detectada,
            "texto_traducido": texto_traducido,
            "confianza": round(confianza, 4),
            "porcentaje": round(confianza * 100, 2),
            "calidad": calidad,
            "mensaje": mensaje,
            "modo": tipo_sena,
            "modo_inicial": tipo_sena_inicial,
            "reevaluado": detalles.get("reevaluado", False),
            "num_frames_procesados": resultado.frames_detectados,
            "num_frames_usados_prediccion": detalles.get("num_frames_usado", reconocedor.num_frames),
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "fuente_keypoints": "cliente" if usar_landmarks else "servidor",
            "modelo_usado": modelo_id,
            "categoria_id": categoria_id,
            "extractor_version": reconocedor.extractor_version,
            "alternativas": alternativas,
            "timestamp": datetime.now().isoformat()
        }

        return RespuestaAPI(
            exito=True,
            mensaje="Reconocimiento completado",
            datos=resultado_final
        )

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
                    'extractor_version': getattr(reconocedor, 'extractor_version', None),
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
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint pendiente"
    )


@router.post("/limpiar-cache")
async def limpiar_cache_traductor(
    modelo_id: Optional[int] = None,
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        mensaje = limpiar_cache(modelo_id)
        return RespuestaAPI(exito=True, mensaje=mensaje)
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return RespuestaAPI(exito=False, mensaje=f"Error: {str(e)}")