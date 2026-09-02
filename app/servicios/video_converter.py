import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoConverter:
    """Conversor robusto de videos para OpenCV"""
    
    @staticmethod
    def convertir_webm_a_frames(ruta_webm: str, output_dir: Path, nombre_sena: str, max_frames: int = 20) -> int:
        """
        Convierte WebM a frames JPG usando métodos robustos.
        Returns: número de frames extraídos
        """
        frames_extraidos = 0
        
        # Método 1: Usar OpenCV con configuración especial
        cap = cv2.VideoCapture(ruta_webm)
        
        # Configurar OpenCV para mejor compatibilidad
        cap.set(cv2.CAP_PROP_FORMAT, -1)
        
        if not cap.isOpened():
            logger.error(f"No se puede abrir el video: {ruta_webm}")
            return 0
        
        try:
            # Intentar leer frames secuencialmente
            frame_count = 0
            while frames_extraidos < max_frames and frame_count < 100:  # Límite de seguridad
                # Método más robusto: usar grab() + retrieve()
                grabbed = cap.grab()
                if not grabbed:
                    break
                
                # Intentar recuperar el frame
                ret, frame = cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    # Guardar cada 2-3 frames para variedad
                    if frame_count % 3 == 0:
                        try:
                            # Procesar frame
                            frame_resized = cv2.resize(frame, (224, 224))
                            
                            # Guardar frame
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
        
        logger.info(f"Extraídos {frames_extraidos} frames de {ruta_webm}")
        return frames_extraidos

    @staticmethod
    def verificar_video_compatible(ruta_video: str) -> bool:
        """Verifica si un video es compatible con OpenCV"""
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            return False
        
        # Intentar leer un frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None