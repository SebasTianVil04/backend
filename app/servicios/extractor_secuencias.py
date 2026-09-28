import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

FEATURES_POR_MANO = 63
FEATURES_TOTALES = 126
VERSION_SECUENCIAS = 3
INDICE_MUÑECA = 0
INDICE_REFERENCIA_ESCALA = 9


@dataclass
class ResultadoSecuencia:
    secuencia: np.ndarray
    estado: np.ndarray
    fps: float
    frames_totales: int
    frames_detectados: int
    keypoints_validos: int
    calidad: float


def crear_detector_manos(
    estatico: bool = False,
    confianza_deteccion: float = 0.5,
    confianza_seguimiento: float = 0.5,
):
    return mp.solutions.hands.Hands(
        static_image_mode=estatico,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=confianza_deteccion,
        min_tracking_confidence=confianza_seguimiento,
    )


def landmarks_de_frame(frame: np.ndarray, hands) -> Tuple[np.ndarray, np.ndarray]:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    resultados = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    fila = np.zeros(FEATURES_TOTALES, dtype=np.float32)
    presencia = np.zeros(2, dtype=bool)

    if resultados.multi_hand_landmarks and resultados.multi_handedness:
        for landmarks, lateralidad in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
            etiqueta = lateralidad.classification[0].label
            slot = 0 if etiqueta == 'Right' else 1
            valores = np.array(
                [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
            ).reshape(-1)
            if valores.size == FEATURES_POR_MANO:
                fila[slot * FEATURES_POR_MANO:(slot + 1) * FEATURES_POR_MANO] = valores
                presencia[slot] = True

    return fila, presencia


def _normalizar_mano(vector63: np.ndarray) -> np.ndarray:
    if not np.any(vector63):
        return vector63

    puntos = vector63.reshape(21, 3).astype(np.float32).copy()
    origen = puntos[INDICE_MUÑECA].copy()
    puntos -= origen

    escala = float(np.linalg.norm(puntos[INDICE_REFERENCIA_ESCALA, :2]))
    if escala < 1e-6:
        escala = 1.0
    puntos /= escala

    return puntos.reshape(-1)


def normalizar_fila(fila: np.ndarray) -> np.ndarray:
    resultado = fila.copy()
    for slot in range(2):
        sl = slice(slot * FEATURES_POR_MANO, (slot + 1) * FEATURES_POR_MANO)
        resultado[sl] = _normalizar_mano(resultado[sl])
    return resultado


def normalizar_filas(filas: np.ndarray) -> np.ndarray:
    normalizado = filas.copy()
    for i in range(len(normalizado)):
        normalizado[i] = normalizar_fila(normalizado[i])
    return normalizado


def _leer_video(ruta_video: str, hands, paso: int):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 500:
        fps = 30.0

    filas = []
    presencias = []
    indice = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if indice % paso == 0:
                fila, presencia = landmarks_de_frame(frame, hands)
                filas.append(fila)
                presencias.append(presencia)
            indice += 1
    finally:
        cap.release()

    if not filas:
        return None

    return np.array(filas, dtype=np.float32), np.array(presencias, dtype=bool), fps, indice


def _interpolar_huecos(valores: np.ndarray, presente: np.ndarray, max_hueco: int):
    resultado = valores.copy()
    interpolado = np.zeros(len(presente), dtype=bool)
    indices = np.flatnonzero(presente)

    for a, b in zip(indices[:-1], indices[1:]):
        hueco = b - a - 1
        if 0 < hueco <= max_hueco:
            t = (np.arange(a + 1, b) - a) / (b - a)
            resultado[a + 1:b] = valores[a] + (valores[b] - valores[a]) * t[:, None]
            interpolado[a + 1:b] = True

    return resultado, interpolado


def construir_secuencia(
    filas: np.ndarray,
    presencias: np.ndarray,
    num_frames: int = 20,
    max_hueco: int = 8,
    fps: float = 30.0,
    frames_totales: Optional[int] = None,
) -> ResultadoSecuencia:
    n = len(filas)
    if frames_totales is None:
        frames_totales = n

    vacio = ResultadoSecuencia(
        secuencia=np.zeros((num_frames, FEATURES_TOTALES), dtype=np.float32),
        estado=np.zeros(num_frames, dtype=np.int8),
        fps=fps,
        frames_totales=frames_totales,
        frames_detectados=0,
        keypoints_validos=0,
        calidad=0.5,
    )

    if n == 0:
        return vacio

    filas = normalizar_filas(filas)

    datos = np.zeros((n, FEATURES_TOTALES), dtype=np.float32)
    estados = np.zeros((n, 2), dtype=np.int8)

    for slot in range(2):
        sl = slice(slot * FEATURES_POR_MANO, (slot + 1) * FEATURES_POR_MANO)
        valores, interpolado = _interpolar_huecos(filas[:, sl], presencias[:, slot], max_hueco)
        datos[:, sl] = valores
        estados[:, slot] = np.where(presencias[:, slot], 2, np.where(interpolado, 1, 0))

    estado_frame = estados.max(axis=1)
    frames_detectados = int(presencias.any(axis=1).sum())
    con_mano = np.flatnonzero(estado_frame > 0)

    if len(con_mano) == 0:
        return vacio

    inicio, fin = con_mano[0], con_mano[-1]
    datos = datos[inicio:fin + 1]
    estado_frame = estado_frame[inicio:fin + 1]

    indices = np.linspace(0, len(datos) - 1, num_frames).round().astype(int)
    secuencia = datos[indices]
    estado = estado_frame[indices]

    keypoints_validos = int((estado == 2).sum())

    detectados = presencias.any(axis=1)
    dos_manos = presencias[detectados].all(axis=1)
    calidad = float(np.where(dos_manos, 0.9, 0.6).mean()) if len(detectados) > 0 else 0.5

    logger.info(
        f"Secuencia: {frames_totales} frames, {frames_detectados} con mano, "
        f"tramo útil {len(datos)} frames, {keypoints_validos}/{num_frames} válidos"
    )

    return ResultadoSecuencia(
        secuencia=secuencia.astype(np.float32),
        estado=estado.astype(np.int8),
        fps=fps,
        frames_totales=frames_totales,
        frames_detectados=frames_detectados,
        keypoints_validos=keypoints_validos,
        calidad=calidad,
    )


def extraer_secuencia_landmarks(
    ruta_video: str,
    num_frames: int = 20,
    paso: int = 1,
    max_hueco: int = 8,
    confianza_deteccion: float = 0.5,
    confianza_seguimiento: float = 0.5,
) -> Optional[ResultadoSecuencia]:
    with crear_detector_manos(False, confianza_deteccion, confianza_seguimiento) as hands:
        lectura = _leer_video(ruta_video, hands, max(1, paso))

    if lectura is None:
        return None

    filas, presencias, fps, frames_totales = lectura
    return construir_secuencia(filas, presencias, num_frames, max_hueco, fps, frames_totales)


def secuencia_desde_frames(
    frames: Sequence[np.ndarray],
    num_frames: int = 20,
    max_hueco: int = 8,
    fps: float = 30.0,
) -> ResultadoSecuencia:
    filas: List[np.ndarray] = []
    presencias: List[np.ndarray] = []

    with crear_detector_manos(
        estatico=True,
        confianza_deteccion=0.4,
        confianza_seguimiento=0.4,
    ) as hands:
        for frame in frames:
            if frame is None:
                continue
            fila, presencia = landmarks_de_frame(frame, hands)
            filas.append(fila)
            presencias.append(presencia)

    if not filas:
        return construir_secuencia(
            np.zeros((0, FEATURES_TOTALES), dtype=np.float32),
            np.zeros((0, 2), dtype=bool),
            num_frames, max_hueco, fps, 0,
        )

    return construir_secuencia(
        np.array(filas, dtype=np.float32), np.array(presencias, dtype=bool),
        num_frames, max_hueco, fps, len(filas),
    )


def secuencia_desde_keypoints(
    keypoints_list: Sequence[np.ndarray],
    num_frames: int = 20,
    max_hueco: int = 8,
    fps: float = 30.0,
) -> ResultadoSecuencia:
    validos = [np.asarray(k, dtype=np.float32).reshape(-1) for k in keypoints_list if k is not None]
    validos = [k for k in validos if k.size == FEATURES_TOTALES]

    if not validos:
        return construir_secuencia(
            np.zeros((0, FEATURES_TOTALES), dtype=np.float32),
            np.zeros((0, 2), dtype=bool),
            num_frames, max_hueco, fps, 0,
        )

    filas = np.stack(validos)
    presencias = np.stack(
        [filas[:, :FEATURES_POR_MANO].any(axis=1), filas[:, FEATURES_POR_MANO:].any(axis=1)],
        axis=1,
    )

    return construir_secuencia(filas, presencias, num_frames, max_hueco, fps, len(filas))


def detectar_lateralidad(resultado: ResultadoSecuencia) -> str:
    if resultado.frames_detectados == 0:
        return "no_detectada"

    mitad_derecha = resultado.secuencia[:, :FEATURES_POR_MANO]
    mitad_izquierda = resultado.secuencia[:, FEATURES_POR_MANO:]

    hay_derecha = bool(np.any(mitad_derecha != 0))
    hay_izquierda = bool(np.any(mitad_izquierda != 0))

    if hay_derecha and hay_izquierda:
        return "ambas"
    if hay_derecha:
        return "derecha"
    if hay_izquierda:
        return "izquierda"
    return "no_detectada"