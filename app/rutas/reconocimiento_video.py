import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual
from ..dependencias.permisos import requiere_permiso
from ..modelos.usuario import Usuario
from ..modelos.entrenamiento import ModeloIA
from ..servicios.gestor_reconocimiento import (
    obtener_reconocedor,
    obtener_reconocedor_por_modelo,
    obtener_reconocedor_cacheado,
    determinar_tipo_sena,
    procesar_frame_base64,
    limpiar_cache,
    cerrar_mediapipe_hands,
    info_cache,
)
from ..servicios.config_tipo_senas import detectar_tipo_sena
from ..servicios.extractor_secuencias import normalizar_fila

router = APIRouter(prefix="/reconocimiento-video", tags=["Reconocimiento Video"])
logger = logging.getLogger(__name__)

FEATURES_POR_MANO = 63
FEATURES_TOTALES = 126


class FrameData(BaseModel):
    frame_base64: str
    timestamp: int
    landmarks: Optional[List[Any]] = None


class SolicitudVideoSecuencia(BaseModel):
    frames: List[FrameData]
    configuracion: Dict[str, Any] = {}
    sena_esperada: Optional[str] = None
    categoria_id: Optional[int] = None
    modelo_id: Optional[int] = None
    tipo_sena: Optional[str] = None


CATEGORIAS_CALIDAD = (
    (0.80, "excelente", "Alta confianza"),
    (0.70, "buena", "Buena confianza"),
    (0.60, "moderada", "Confianza moderada"),
    (0.50, "baja", "Baja confianza"),
)

MAX_FRAMES_ENTRADA = 120
MIN_FRAMES_CON_MANO = 4
TIPOS_SENA = ("DINAMICA", "ESTATICA")

# --- Parámetros del modo automático (provisionales: calibrar con las líneas de diagnóstico) ---
# El movimiento se mide como desplazamiento medio por paso entre frames, en "tamaños de mano",
# dentro de la ventana MÁS QUIETA de la captura (así entrar y salir con la mano no cuenta).
VENTANA_QUIETA = 0.4             # fracción de la captura que forma la ventana
UMBRAL_QUIETA_DINAMICA = 0.08    # >= esto en la ventana más quieta => movimiento sostenido
UMBRAL_QUIETA_ESTATICA = 0.04    # <= esto => la mano se quedó quieta en algún tramo
CONF_SATURADA = 0.90             # por encima, el softmax ya no distingue entre cabezas
DIF_CONF_DECISIVA = 0.15         # si una cabeza gana por más que esto, manda la confianza
CONFIANZA_MINIMA_PREFERIDO = 0.35


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


def _xy(punto) -> Tuple[float, float]:
    if isinstance(punto, dict):
        return float(punto.get('x', 0.0) or 0.0), float(punto.get('y', 0.0) or 0.0)
    return float(getattr(punto, 'x', 0.0) or 0.0), float(getattr(punto, 'y', 0.0) or 0.0)


def _minimo_ventana(pasos: np.ndarray, fraccion: float) -> float:
    # Promedio de movimiento de la ventana contigua más quieta.
    n = len(pasos)
    if n == 0:
        return 0.0
    w = min(n, max(2, int(round(n * fraccion))))
    medias = np.convolve(pasos, np.ones(w, dtype=np.float32) / w, mode="valid")
    return float(medias.min())


def _movimiento_manos(frames: List[FrameData]) -> Dict[str, float]:
    # Devuelve el movimiento de las manos en "tamaños de mano" por paso entre frames:
    #  - quieta: movimiento medio en la ventana más quieta (muñeca + dedos). Es el valor de decisión.
    #  - forma_quieta: lo mismo pero solo la forma de la mano (relativa a la muñeca).
    #  - medio / pico: movimiento medio y máximo de toda la captura (informativos).
    # Con varias manos se toma la que más se mueve.
    series: List[List[np.ndarray]] = [[], []]

    for frame in frames:
        manos = frame.landmarks
        if not manos or not isinstance(manos, (list, tuple)):
            continue
        for slot, mano in enumerate(manos[:2]):
            if not isinstance(mano, (list, tuple)) or len(mano) < 21:
                continue
            series[slot].append(np.asarray([_xy(p) for p in mano[:21]], dtype=np.float32))

    resultado = {"quieta": 0.0, "forma_quieta": 0.0, "medio": 0.0, "pico": 0.0, "frames_mano": 0}

    for serie in series:
        if len(serie) < MIN_FRAMES_CON_MANO:
            continue
        arr = np.stack(serie)                                        # (n, 21, 2)
        extension = arr.max(axis=1) - arr.min(axis=1)                # (n, 2)
        escala = float(np.median(np.linalg.norm(extension, axis=1)))  # tamaño típico de la mano
        if escala < 1e-4:
            continue
        arr = arr / escala
        rel = arr - arr[:, :1, :]

        pasos = np.linalg.norm(np.diff(arr, axis=0), axis=2).mean(axis=1)
        pasos_forma = np.linalg.norm(np.diff(rel, axis=0), axis=2).mean(axis=1)

        resultado["quieta"] = max(resultado["quieta"], _minimo_ventana(pasos, VENTANA_QUIETA))
        resultado["forma_quieta"] = max(resultado["forma_quieta"], _minimo_ventana(pasos_forma, VENTANA_QUIETA))
        resultado["medio"] = max(resultado["medio"], float(pasos.mean()))
        resultado["pico"] = max(resultado["pico"], float(pasos.max()))
        resultado["frames_mano"] = max(resultado["frames_mano"], len(serie))

    return resultado


def _evaluar_calidad(confianza: float, sena_detectada: str) -> Dict[str, str]:
    for umbral, calidad, etiqueta in CATEGORIAS_CALIDAD:
        if confianza >= umbral:
            return {"calidad": calidad, "mensaje": f"{etiqueta}: {sena_detectada}"}
    return {"calidad": "muy_baja", "mensaje": "Seña no reconocida claramente"}


def _respuesta_fallida(mensaje, categoria_id, frames_procesados=0, errores=None):
    r = {
        "exito": False,
        "mensaje": mensaje,
        "datos": {
            "confianza": 0.0,
            "sena_detectada": "",
            "num_frames_procesados": frames_procesados,
            "categoria_id": categoria_id
        }
    }
    if errores:
        r["errores"] = errores
    return r


def _iso(valor) -> Optional[str]:
    return valor.isoformat() if valor else None


def _tipo_por_nombre(sena: str, reconocedor, tipo_defecto: str) -> str:
    # Prioridad: el tipo con el que se entrenó la clase; la heurística por nombre solo es respaldo.
    tipo = None
    try:
        tipo = (getattr(reconocedor, "tipos_senas", None) or {}).get(sena)
    except Exception:
        tipo = None
    if not tipo:
        try:
            tipo = detectar_tipo_sena(sena)
        except Exception:
            tipo = None
    tipo_final = str(tipo or tipo_defecto).upper()
    return "DINAMICA" if tipo_final == "EMERGENCIA" else tipo_final


def _logits_cabeza(reconocedor, tensor, tipo: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        with torch.no_grad():
            salida = reconocedor._ejecutar_forward(tensor, [tipo])
        valores = np.sort(salida[0].float().cpu().numpy())[::-1]
        margen = float(valores[0] - valores[1]) if len(valores) > 1 else 0.0
        return float(valores[0]), margen
    except Exception:
        return None, None


def _predecir_probando_ambos_tipos(
    reconocedor,
    tensor,
    movimiento: Optional[Dict[str, float]] = None,
) -> Tuple[str, float, Dict[str, Any], str]:
    movimiento = movimiento or {}
    candidatos = []

    for tipo in TIPOS_SENA:
        sena, conf, detalles = reconocedor.predecir(tensor, tipo, None)
        if not sena or sena == "desconocido":
            continue

        tipo_nombre = _tipo_por_nombre(sena, reconocedor, tipo)
        logit_max, margen_logit = _logits_cabeza(reconocedor, tensor, tipo)
        candidatos.append({
            "coherente": tipo_nombre == tipo,
            "confianza": conf,
            "tipo": tipo,
            "sena": sena,
            "tipo_modelo": reconocedor.tipos_senas.get(sena),
            "logit_max": logit_max,
            "margen_logit": margen_logit,
            "score": logit_max if logit_max is not None else 0.0,
            "detalles": detalles,
        })

    if not candidatos:
        return "desconocido", 0.0, {"error": "No se pudo reconocer la seña", "alternativas": []}, "DINAMICA"

    # 1) Qué tipo sugiere el movimiento (en la ventana más quieta de la captura)
    quieta = float(movimiento.get("quieta", 0.0))
    if quieta >= UMBRAL_QUIETA_DINAMICA:
        tipo_favorecido = "DINAMICA"
    elif quieta <= UMBRAL_QUIETA_ESTATICA:
        tipo_favorecido = "ESTATICA"
    else:
        tipo_favorecido = None

    elegido = None
    motivo = ""

    # 2) Si el movimiento es claro, decide el movimiento (las cabezas saturan al 100% y no sirven)
    if tipo_favorecido:
        del_tipo = [
            c for c in candidatos
            if c["tipo"] == tipo_favorecido and c["confianza"] >= CONFIANZA_MINIMA_PREFERIDO
        ]
        if del_tipo:
            elegido = max(del_tipo, key=lambda c: c["confianza"])
            motivo = "movimiento"

    # 3) Si es ambiguo: confianza si hay diferencia clara; si no, logit máximo
    if elegido is None:
        coherentes = [c for c in candidatos if c["coherente"]]
        pool = coherentes or candidatos
        ordenados = sorted(pool, key=lambda c: c["confianza"], reverse=True)
        elegido = ordenados[0]
        motivo = "confianza"
        if len(ordenados) > 1:
            segundo = ordenados[1]
            ambos_saturados = (
                elegido["confianza"] >= CONF_SATURADA and segundo["confianza"] >= CONF_SATURADA
            )
            if ambos_saturados or (elegido["confianza"] - segundo["confianza"]) < DIF_CONF_DECISIVA:
                elegido = max(pool, key=lambda c: c["score"])
                motivo = "logit"

    detalles = elegido["detalles"]

    alternativas = list(detalles.get("alternativas", []))
    for otro in candidatos:
        if otro is elegido or otro["sena"] == elegido["sena"]:
            continue
        alternativas = [a for a in alternativas if a.get("sena") != otro["sena"]]
        alternativas.append({
            "sena": otro["sena"],
            "confianza": otro["confianza"],
            "tipo": otro["tipo"],
        })
    detalles["alternativas"] = alternativas

    detalles["diagnostico_auto"] = {
        "movimiento_quieto": round(quieta, 4),
        "forma_quieta": round(float(movimiento.get("forma_quieta", 0.0)), 4),
        "movimiento_medio": round(float(movimiento.get("medio", 0.0)), 4),
        "movimiento_pico": round(float(movimiento.get("pico", 0.0)), 4),
        "umbral_dinamica": UMBRAL_QUIETA_DINAMICA,
        "umbral_estatica": UMBRAL_QUIETA_ESTATICA,
        "tipo_favorecido": tipo_favorecido,
        "motivo": motivo,
        "tipo_elegido": elegido["tipo"],
        "candidatos": [
            {
                "tipo": c["tipo"],
                "sena": c["sena"],
                "confianza": round(float(c["confianza"]), 3),
                "coherente": c["coherente"],
                "tipo_modelo": c["tipo_modelo"],
                "logit_max": round(c["logit_max"], 2) if c["logit_max"] is not None else None,
                "margen_logit": round(c["margen_logit"], 2) if c["margen_logit"] is not None else None,
            }
            for c in candidatos
        ],
    }
    detalles["reevaluado"] = True
    detalles["tipo_inicial"] = "AUTO"

    logger.info(
        f"[AUTO-TIPO] quieta={quieta:.4f} (estatica<={UMBRAL_QUIETA_ESTATICA}, dinamica>={UMBRAL_QUIETA_DINAMICA}) "
        f"medio={movimiento.get('medio', 0.0):.4f} forma_quieta={movimiento.get('forma_quieta', 0.0):.4f} "
        f"favorecido={tipo_favorecido} motivo={motivo} | " + " | ".join(
            f"{c['tipo']}->{c['sena']} ({c['confianza'] * 100:.1f}%, coherente={c['coherente']}, "
            f"tipo_modelo={c['tipo_modelo']}, logit={c['logit_max']})"
            for c in candidatos
        ) + f" => {elegido['tipo']}:{elegido['sena']}"
    )

    return elegido["sena"], elegido["confianza"], detalles, elegido["tipo"]


@router.get("/modelos-activos")
def listar_modelos_activos(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        activos = db.query(ModeloIA).filter(ModeloIA.activo == True).all()

        disponibles = [
            m for m in activos
            if m.ruta_archivo and os.path.exists(m.ruta_archivo)
        ]
        disponibles.sort(key=lambda m: m.fecha_entrenamiento or datetime.min, reverse=True)

        modelos = [
            {
                "id": m.id,
                "nombre": m.nombre,
                "accuracy": m.accuracy or 0.0,
                "num_clases": m.num_clases or 0,
                "arquitectura": getattr(m, "arquitectura", None),
                "tipo": getattr(m, "tipo_modelo", None),
                "fecha_activacion": _iso(getattr(m, "fecha_activacion", None)),
            }
            for m in disponibles
        ]

        return {
            "exito": True,
            "mensaje": f"{len(modelos)} modelos activos",
            "datos": {
                "total": len(modelos),
                "modelos": modelos,
            }
        }

    except Exception as e:
        logger.error(f"Error listando modelos activos: {e}", exc_info=True)
        return {
            "exito": False,
            "mensaje": f"Error: {e}",
            "datos": {"total": 0, "es_ensemble": False, "modelos": []}
        }


@router.post("/secuencia", response_model=Dict[str, Any])
def reconocer_secuencia_video(
    solicitud: SolicitudVideoSecuencia,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    start_time = time.time()

    try:
        num_frames_recibidos = len(solicitud.frames)

        if num_frames_recibidos == 0:
            return _respuesta_fallida("No se recibieron frames", solicitud.categoria_id)

        if num_frames_recibidos > MAX_FRAMES_ENTRADA:
            return _respuesta_fallida(
                f"Demasiados frames ({num_frames_recibidos}, máx {MAX_FRAMES_ENTRADA})",
                solicitud.categoria_id
            )

        try:
            if solicitud.modelo_id:
                reconocedor, modelo_id = obtener_reconocedor_por_modelo(db, solicitud.modelo_id)
            else:
                reconocedor, modelo_id = obtener_reconocedor(db, solicitud.categoria_id)
        except ValueError as e:
            return _respuesta_fallida(str(e), solicitud.categoria_id)

        keypoints_cliente: List[np.ndarray] = []
        for frame_data in solicitud.frames:
            if frame_data.landmarks:
                fila = _landmarks_a_fila(frame_data.landmarks)
                if fila is not None:
                    keypoints_cliente.append(fila)

        usar_landmarks = len(keypoints_cliente) >= MIN_FRAMES_CON_MANO

        if usar_landmarks:
            resultado_secuencia = reconocedor._resultado_desde_keypoints(keypoints_cliente)
        else:
            frames_validos: List[np.ndarray] = []
            for frame_data in solicitud.frames:
                frame = procesar_frame_base64(frame_data.frame_base64)
                if frame is not None:
                    frames_validos.append(frame)

            if len(frames_validos) < MIN_FRAMES_CON_MANO:
                return _respuesta_fallida(
                    f"Frames inválidos ({len(frames_validos)}/{MIN_FRAMES_CON_MANO})",
                    solicitud.categoria_id, len(frames_validos)
                )
            resultado_secuencia = reconocedor._resultado_desde_frames(frames_validos)

        frames_exitosos = resultado_secuencia.frames_detectados

        if frames_exitosos < MIN_FRAMES_CON_MANO:
            return _respuesta_fallida(
                f"Frames con mano insuficientes ({frames_exitosos}/{MIN_FRAMES_CON_MANO})",
                solicitud.categoria_id, frames_exitosos
            )

        keypoints_secuencia = [
            fila for fila, estado in zip(resultado_secuencia.secuencia, resultado_secuencia.estado)
            if estado > 0
        ]

        tensor = reconocedor._tensor_desde_resultado(resultado_secuencia)

        tipo_forzado = (solicitud.tipo_sena or "").strip().upper()

        if tipo_forzado in TIPOS_SENA:
            tipo_sena_inicial = tipo_forzado
            sena_detectada, confianza, detalles = reconocedor.predecir(
                tensor, tipo_sena_inicial, solicitud.sena_esperada
            )
            tipo_sena = detalles.get("tipo_sena", tipo_sena_inicial)

        elif solicitud.sena_esperada:
            tipo_sena_inicial = determinar_tipo_sena(
                db, solicitud.sena_esperada, solicitud.categoria_id, reconocedor,
                keypoints_secuencia=keypoints_secuencia
            )
            sena_detectada, confianza, detalles = reconocedor.predecir(
                tensor, tipo_sena_inicial, solicitud.sena_esperada
            )
            tipo_sena = detalles.get("tipo_sena", tipo_sena_inicial)

        else:
            movimiento = _movimiento_manos(solicitud.frames)
            sena_detectada, confianza, detalles, tipo_sena = _predecir_probando_ambos_tipos(
                reconocedor, tensor, movimiento
            )
            tipo_sena_inicial = "AUTO"

        if not sena_detectada or sena_detectada == "desconocido":
            return _respuesta_fallida(
                detalles.get("error") or "No se pudo reconocer la seña",
                solicitud.categoria_id, frames_exitosos
            )

        confianza_validada = max(0.0, min(1.0, confianza))
        processing_time = time.time() - start_time
        evaluacion = _evaluar_calidad(confianza_validada, sena_detectada)

        alternativas_filtradas = sorted(
            [alt for alt in detalles.get("alternativas", []) if alt.get("confianza", 0) >= 0.2],
            key=lambda x: x.get("confianza", 0),
            reverse=True
        )[:3]

        es_ensemble = bool(getattr(reconocedor, "es_ensemble", False))

        resultado = {
            "sena_detectada": sena_detectada,
            "texto_traducido": sena_detectada,
            "confianza": round(confianza_validada, 4),
            "confianza_raw": round(confianza_validada, 4),
            "porcentaje": round(confianza_validada * 100, 2),
            "calidad": evaluacion["calidad"],
            "mensaje": evaluacion["mensaje"],
            "modo": tipo_sena,
            "modo_inicial": tipo_sena_inicial,
            "reevaluado": detalles.get("reevaluado", False),
            "diagnostico_auto": detalles.get("diagnostico_auto"),
            "num_frames_procesados": frames_exitosos,
            "num_frames_usados_prediccion": detalles.get("num_frames_usado", reconocedor.num_frames),
            "tiempo_procesamiento_ms": round(processing_time * 1000, 2),
            "fps": round(num_frames_recibidos / processing_time, 2) if processing_time > 0 else 0,
            "fuente_keypoints": "cliente" if usar_landmarks else "servidor",
            "alternativas": alternativas_filtradas,
            "modelo_usado": {
                "modelo_id": modelo_id,
                "categoria_id": solicitud.categoria_id,
                "modelo_nombre": reconocedor.arquitectura,
                "modo_usado": tipo_sena,
                "es_ensemble": es_ensemble,
                "ensemble": detalles.get("modelos_ensemble"),
                "seleccion_manual": bool(solicitud.modelo_id),
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"Reconocimiento [Cat:{solicitud.categoria_id}] [Modelo:{modelo_id}"
            f"{' (manual)' if solicitud.modelo_id else ''}] "
            f"[keypoints={resultado['fuente_keypoints']}] [ensemble={es_ensemble}]: {sena_detectada} "
            f"({confianza_validada * 100:.1f}%) - Modo: {tipo_sena}"
        )

        return {"exito": True, "mensaje": "Reconocimiento completado", "datos": resultado}

    except Exception as e:
        logger.error(f"Error en reconocimiento: {e}", exc_info=True)
        return _respuesta_fallida(
            "Error en el reconocimiento", solicitud.categoria_id, errores=[f"Error interno: {e}"]
        )


@router.get("/info-modelo")
def obtener_info_modelo(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        reconocedor, modelo_id = obtener_reconocedor(db)
        modelo_db = db.query(ModeloIA).filter(ModeloIA.id == modelo_id).first()
        es_ensemble = bool(getattr(reconocedor, "es_ensemble", False))

        miembros = None
        if es_ensemble:
            miembros = [
                {"modelo_id": m.modelo_id, "nombre": m.nombre, "peso": m.peso,
                 "accuracy": round((m.reconocedor.accuracy or 0.0) * 100, 2)}
                for m in reconocedor.miembros
            ]

        return {
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
                "tipos_senas": reconocedor.tipos_senas,
                "extractor_version": getattr(reconocedor, "extractor_version", None),
                "es_ensemble": es_ensemble,
                "miembros_ensemble": miembros,
            },
            "sistema": {
                "device": str(reconocedor.device),
                "cuda_disponible": str(reconocedor.device).startswith('cuda'),
                "amp_enabled": reconocedor.enable_amp,
                "modo": "ensemble_keypoints_lstm" if es_ensemble else "keypoints_lstm"
            }
        }

    except Exception as e:
        logger.error(f"Error obteniendo info: {e}")
        return {"exito": False, "mensaje": f"Error: {e}"}


@router.get("/estado")
def obtener_estado_reconocimiento(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    try:
        cache_info = info_cache()
        reconocedor = None
        modelo_id = None

        if cache_info['modelos_cargados']:
            modelo_id = cache_info['modelos_cargados'][0]
            reconocedor = obtener_reconocedor_cacheado(modelo_id)
        else:
            try:
                reconocedor, modelo_id = obtener_reconocedor(db)
            except Exception as e:
                logger.warning(f"No se pudo cargar reconocedor: {e}")

        return {
            "estado": "activo" if reconocedor is not None else "inactivo",
            "modelo_cargado": reconocedor is not None,
            "arquitectura": reconocedor.arquitectura if reconocedor else "ninguna",
            "clases_cargadas": len(reconocedor.clases) if reconocedor else 0,
            "clases": reconocedor.clases if reconocedor else [],
            "num_frames": reconocedor.num_frames if reconocedor else 0,
            "device": str(reconocedor.device) if reconocedor else "ninguno",
            "ultimo_uso": cache_info['ultimo_acceso'].get(modelo_id) if modelo_id else None,
            "modo": "keypoints_lstm",
            "modelos_en_cache": cache_info['modelos_cargados'],
            "ensemble_activo": cache_info.get('ensemble_activo', False),
            "ensemble_modelos": cache_info.get('ensemble_modelos', []),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return {"estado": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


@router.post("/limpiar-cache")
def limpiar_cache_modelo(
    usuario_actual: Usuario = Depends(requiere_permiso("modelos.gestionar"))
):
    try:
        mensaje = limpiar_cache()
        cerrar_mediapipe_hands()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Cache del modelo limpiado")
        return {"exito": True, "mensaje": mensaje}
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        return {"exito": False, "mensaje": f"Error: {e}"}