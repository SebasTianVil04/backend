import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
import logging

from app.servicios.config_tipo_senas import obtener_config_sena

logger = logging.getLogger(__name__)


class VideoConverter:

    @staticmethod
    def convertir_webm_a_frames(ruta_webm: str, output_dir: Path, nombre_sena: str, max_frames: int = None) -> int:
        if max_frames is None:
            config = obtener_config_sena(nombre_sena)
            max_frames = config.get('num_frames_recomendado', 20)

        frames_extraidos = 0

        cap = cv2.VideoCapture(ruta_webm)

        if not cap.isOpened():
            logger.error(f"No se puede abrir el video: {ruta_webm}")
            return 0

        try:
            frame_count = 0
            max_iteraciones = max(100, max_frames * 8)
            intervalo_guardado = 3

            while frames_extraidos < max_frames and frame_count < max_iteraciones:
                grabbed = cap.grab()
                if not grabbed:
                    break

                ret, frame = cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    if frame_count % intervalo_guardado == 0:
                        try:
                            frame_resized = cv2.resize(frame, (224, 224))

                            frame_filename = f"{nombre_sena}_{frames_extraidos:04d}.jpg"
                            frame_path = output_dir / frame_filename

                            if cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                                frames_extraidos += 1
                                logger.debug(f"Frame {frames_extraidos} guardado exitosamente")

                        except Exception as e:
                            logger.warning(f"Error guardando frame {frame_count}: {e}")

                frame_count += 1

        except Exception as e:
            logger.error(f"Error durante la extracción: {e}")

        finally:
            cap.release()

        logger.info(f"Extraídos {frames_extraidos}/{max_frames} frames de {ruta_webm}")
        return frames_extraidos

    @staticmethod
    def verificar_video_compatible(ruta_video: str) -> bool:
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()

        return ret and frame is not None