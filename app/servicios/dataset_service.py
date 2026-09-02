import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import tempfile
import gc
import subprocess
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging
import time
import mediapipe as mp

from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..servicios.config_tipo_senas import detectar_tipo_sena, obtener_config_sena
from ..servicios.drive_service import subir_archivo_a_drive

logger = logging.getLogger(__name__)

class DatasetService:
    
    def __init__(self):
        self.video_dir = Path("archivos_subidos") / "videos_dataset"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        self.resolucion_frame = (224, 224)
        self.ffmpeg_disponible = self._verificar_ffmpeg()
        
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info(f"DatasetService inicializado - FFmpeg: {'Disponible' if self.ffmpeg_disponible else 'No disponible'}")

    def _verificar_ffmpeg(self) -> bool:
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def _convertir_webm_a_mp4(self, ruta_webm: str, ruta_mp4: str) -> bool:
        if not self.ffmpeg_disponible:
            logger.warning("FFmpeg no disponible, saltando conversión")
            return False
        
        try:
            cmd = [
                'ffmpeg',
                '-i', ruta_webm,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '28',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-vf', 'fps=30',
                '-an',
                '-y',
                ruta_mp4
            ]
            
            logger.info(f"Convirtiendo WebM a MP4: {os.path.basename(ruta_webm)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and os.path.exists(ruta_mp4):
                tamaño_mp4 = os.path.getsize(ruta_mp4)
                if tamaño_mp4 > 1024:
                    logger.info(f"✓ Conversión exitosa: {tamaño_mp4:,} bytes")
                    return True
                else:
                    logger.error(f"✗ MP4 muy pequeño: {tamaño_mp4} bytes")
                    if os.path.exists(ruta_mp4):
                        os.remove(ruta_mp4)
                    return False
            else:
                logger.error(f"✗ FFmpeg falló con código {result.returncode}")
                if result.stderr:
                    logger.error(f"Error FFmpeg: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("✗ Timeout en conversión FFmpeg (60s)")
            return False
        except Exception as e:
            logger.error(f"✗ Error en conversión: {e}")
            return False

    def _extraer_keypoints_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(frame_rgb)
            
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

    def _extraer_keypoints_opencv_robusto(self, ruta_video: str, nombre_sena: str, max_frames: int = 20) -> Tuple[int, float]:
        backends = [
            ('CAP_FFMPEG', cv2.CAP_FFMPEG),
            ('CAP_ANY (default)', cv2.CAP_ANY),
        ]
        
        for backend_name, backend_flag in backends:
            logger.info(f"🔍 Intentando extraer keypoints con {backend_name}...")
            resultado = self._extraer_keypoints_con_backend(
                ruta_video, nombre_sena, max_frames, backend_flag, backend_name
            )
            
            if resultado[0] > 0:
                logger.info(f"✓ Extracción exitosa con {backend_name}: {resultado[0]} keypoints")
                return resultado
            else:
                logger.warning(f"⚠ Falló con {backend_name}, intentando siguiente método...")
        
        logger.error(f"✗ No se pudieron extraer keypoints con ningún método")
        return 0, 0.0

    def _extraer_keypoints_con_backend(
        self, 
        ruta_video: str, 
        nombre_sena: str, 
        max_frames: int,
        backend: int,
        backend_name: str
    ) -> Tuple[int, float]:
        """
        Analiza el video y cuenta cuántos frames tienen manos detectables,
        calculando una calidad promedio. Ya NO guarda los keypoints en disco
        (los .npy no se usan en ningún otro punto del sistema: ni entrenamiento
        ni reconocimiento leen estos archivos, ambos recalculan los keypoints
        directamente desde el video cuando los necesitan).
        """
        cap = None
        keypoints_detectados = 0
        calidad_total = 0.0
        
        try:
            cap = cv2.VideoCapture(ruta_video, backend)
            
            if not cap.isOpened():
                return 0, 0.0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps is None or fps <= 0 or fps > 500:
                fps = 30.0
            
            logger.info(f"   Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")
            
            if width <= 0 or height <= 0 or width > 10000 or height > 10000:
                logger.warning(f"   ⚠ Dimensiones inválidas: {width}x{height}")
                return 0, 0.0
            
            frame_count = 0
            frames_saltados = max(1, 3)
            intentos_fallidos_consecutivos = 0
            max_intentos_consecutivos = 10
            max_iteraciones = 150
            
            while keypoints_detectados < max_frames and frame_count < max_iteraciones:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    intentos_fallidos_consecutivos += 1
                    if intentos_fallidos_consecutivos >= max_intentos_consecutivos:
                        logger.warning(f"   ⚠ {max_intentos_consecutivos} fallos consecutivos, deteniendo")
                        break
                    frame_count += 1
                    continue
                
                if frame_count % frames_saltados != 0:
                    frame_count += 1
                    continue
                
                if len(frame.shape) != 3:
                    logger.warning(f"   ⚠ Shape inválido: {frame.shape}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                frame_height, frame_width, frame_channels = frame.shape
                
                if frame_height < 10 or frame_width < 10:
                    logger.warning(f"   ⚠ Frame muy pequeño: {frame_width}x{frame_height}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                if frame_height == 1 or frame_width == 1:
                    logger.warning(f"   ⚠ Frame corrupto (dimensión = 1): {frame_width}x{frame_height}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                if frame_height > 10000 or frame_width > 10000:
                    logger.warning(f"   ⚠ Frame corrupto (muy grande): {frame_width}x{frame_height}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                if frame_channels not in [1, 3, 4]:
                    logger.warning(f"   ⚠ Canales inválidos: {frame_channels}")
                    intentos_fallidos_consecutivos += 1
                    frame_count += 1
                    continue
                
                intentos_fallidos_consecutivos = 0
                
                try:
                    if frame_channels == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame_channels == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    frame_resized = cv2.resize(frame, self.resolucion_frame, interpolation=cv2.INTER_AREA)
                    
                    keypoints = self._extraer_keypoints_frame(frame_resized)
                    
                    if keypoints is not None:
                        # Ya no se guarda en disco (np.save eliminado):
                        # estos keypoints solo se usan aquí para calcular
                        # calidad/cantidad; el entrenamiento y el reconocimiento
                        # los recalculan directo desde el video cuando los necesitan.
                        keypoints_detectados += 1
                        
                        num_manos = 1 if np.sum(keypoints[63:]) == 0 else 2
                        calidad = 0.9 if num_manos == 2 else 0.6
                        calidad_total += calidad
                        
                        if keypoints_detectados % 5 == 0:
                            logger.info(f"   ✓ {keypoints_detectados}/{max_frames} keypoints detectados")
                    
                except Exception as e:
                    logger.warning(f"   ⚠ Error procesando frame: {e}")
                    intentos_fallidos_consecutivos += 1
                
                frame_count += 1
            
            calidad_promedio = calidad_total / keypoints_detectados if keypoints_detectados > 0 else 0.5
            return keypoints_detectados, calidad_promedio
            
        except Exception as e:
            logger.error(f"   ✗ Error con {backend_name}: {e}")
            return 0, 0.0
        finally:
            if cap is not None:
                cap.release()

    def _procesar_video_directo(self, ruta_video: str, nombre_sena: str) -> Tuple[int, float, float, int, float]:
        cap = None
        
        try:
            keypoints_extraidos, calidad_promedio = self._extraer_keypoints_opencv_robusto(ruta_video, nombre_sena)
            
            cap = cv2.VideoCapture(ruta_video)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps is None or fps <= 0 or fps > 500:
                    fps = 30.0
                
                if total_frames <= 0 or total_frames > 1000000:
                    total_frames = max(keypoints_extraidos * 2, 30)
                
                duracion = total_frames / fps if fps > 0 else 1.0
                
                cap.release()
            else:
                fps = 30.0
                total_frames = keypoints_extraidos * 2
                duracion = 1.0
            
            return keypoints_extraidos, calidad_promedio, fps, total_frames, duracion
            
        except Exception as e:
            logger.error(f"✗ Error procesando video: {e}", exc_info=True)
            return 0, 0.5, 30.0, 0, 1.0
        finally:
            if cap is not None:
                cap.release()

    def _subir_video_a_drive_y_limpiar(self, ruta_video_local: str, formato: str) -> Dict[str, Optional[str]]:
        """
        Sube el video a Google Drive. Si tiene éxito, borra el archivo local
        (ya no necesitamos guardarlo en disco permanentemente).
        Si falla, mantiene el archivo local como respaldo y retorna
        drive_file_id=None para que el registro en BD sepa que aún no se subió.
        """
        resultado = {"drive_file_id": None, "drive_url": None}
        ruta = Path(ruta_video_local)

        if not ruta.exists():
            logger.warning(f"[Drive] Archivo no encontrado: {ruta_video_local}")
            return resultado

        try:
            mime_type = "video/mp4" if formato == "mp4" else "video/webm"

            with open(ruta, "rb") as f:
                contenido = f.read()

            subida = subir_archivo_a_drive(
                contenido,
                ruta.name,
                mime_type,
                subcarpeta="videos_dataset",
            )

            resultado["drive_file_id"] = subida["id"]
            resultado["drive_url"] = subida["url_ver"]

            logger.info(f"[Drive] Subido correctamente: {ruta.name} -> {subida['url_ver']}")

            try:
                ruta.unlink()
                logger.info(f"[Drive] Archivo local eliminado: {ruta.name}")
            except Exception as e_del:
                logger.warning(f"[Drive] No se pudo eliminar el archivo local (no crítico): {e_del}")

        except Exception as e:
            logger.warning(f"[Drive] Falló la subida, se mantiene el archivo local como respaldo: {e}")

        return resultado

    async def subir_video_dataset(
        self,
        db: Session,
        archivo: any,
        categoria_id: int,
        sena: str,
        usuario_id: int
    ) -> VideoDataset:
        temp_video_path = None
        temp_mp4_path = None
        
        try:
            if hasattr(archivo, 'read'):
                contenido_video = await archivo.read()
            else:
                contenido_video = archivo
            
            if len(contenido_video) < 1024:
                raise ValueError("Archivo de video demasiado pequeño")
            
            logger.info(f"📹 Procesando video: {sena}, {len(contenido_video):,} bytes")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            nombre_webm = f"{sena}_{timestamp}.webm"
            ruta_webm_final = self.video_dir / nombre_webm
            
            with open(ruta_webm_final, 'wb') as f:
                f.write(contenido_video)
            
            logger.info(f"✓ WebM guardado: {ruta_webm_final}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                temp_file.write(contenido_video)
                temp_video_path = temp_file.name
            
            conversion_exitosa = False
            formato_guardado = "webm"
            ruta_video_final = str(ruta_webm_final)
            
            if self.ffmpeg_disponible:
                nombre_mp4 = f"{sena}_{timestamp}.mp4"
                ruta_mp4_final = self.video_dir / nombre_mp4
                temp_mp4_path = temp_video_path.replace('.webm', '_temp.mp4')
                
                logger.info(f"🔄 Convirtiendo a MP4...")
                
                if self._convertir_webm_a_mp4(temp_video_path, temp_mp4_path):
                    if os.path.exists(temp_mp4_path) and os.path.getsize(temp_mp4_path) > 1024:
                        import shutil
                        shutil.move(temp_mp4_path, str(ruta_mp4_final))
                        conversion_exitosa = True
                        formato_guardado = "mp4"
                        ruta_video_final = str(ruta_mp4_final)
                        
                        logger.info(f"✓ MP4 guardado exitosamente: {ruta_mp4_final}")
                        
                        try:
                            os.remove(ruta_webm_final)
                            logger.info("✓ WebM eliminado (MP4 es la versión final)")
                        except Exception as e:
                            logger.warning(f"No se pudo eliminar WebM: {e}")
                    else:
                        logger.warning("⚠ MP4 generado es inválido, intentando con WebM")
                else:
                    logger.warning("⚠ Conversión a MP4 falló, intentando con WebM")
            else:
                logger.warning("⚠ FFmpeg no disponible, intentando procesar WebM directamente")
                logger.warning("   RECOMENDACIÓN: Instale FFmpeg para mejor compatibilidad")
            
            logger.info(f"📊 Extrayendo keypoints de {formato_guardado.upper()}...")
            keypoints_extraidos, calidad_promedio, fps_real, total_frames, duracion_real = self._procesar_video_directo(
                ruta_video_final, sena
            )
            
            aprobado = keypoints_extraidos >= 3

            # --- Subir a Drive y limpiar local (ya no se guarda copia permanente en disco) ---
            drive_info = self._subir_video_a_drive_y_limpiar(ruta_video_final, formato_guardado)
            subio_a_drive = drive_info["drive_file_id"] is not None

            # Si subió a Drive, la ruta local ya no existe; guardamos None.
            # Si falló, dejamos la ruta local como respaldo para poder reintentar después.
            ruta_para_bd = None if subio_a_drive else ruta_video_final

            notas_extra = ""
            if not subio_a_drive:
                notas_extra = " | ADVERTENCIA: no se pudo subir a Drive, archivo conservado localmente"

            video_db = VideoDataset(
                categoria_id=categoria_id,
                usuario_id=usuario_id,
                sena=sena.upper(),
                ruta_video=ruta_para_bd,
                drive_file_id=drive_info["drive_file_id"],
                drive_url=drive_info["drive_url"],
                duracion_segundos=duracion_real,
                fps=fps_real,
                resolucion=json.dumps({"ancho": 640, "alto": 480}),
                tamaño_bytes=len(contenido_video),
                formato=formato_guardado,
                frames_extraidos=keypoints_extraidos,
                calidad_promedio=calidad_promedio,
                procesado=True,
                aprobado=aprobado,
                fecha_procesado=datetime.utcnow(),
                notas=f"Formato: {formato_guardado.upper()}, Keypoints: {keypoints_extraidos}, Calidad: {calidad_promedio:.2f}{notas_extra}"
            )
            
            if aprobado:
                video_db.fecha_aprobado = datetime.utcnow()

            db.add(video_db)
            db.commit()
            db.refresh(video_db)

            estado = "✓ APROBADO" if aprobado else "⚠ PENDIENTE"
            drive_estado = "☁ Drive OK" if subio_a_drive else "⚠ Solo local"
            logger.info(f"{estado} - Video {video_db.id} - {keypoints_extraidos} keypoints - {formato_guardado.upper()} - {drive_estado}")

            return video_db

        except Exception as e:
            db.rollback()
            logger.error(f"✗ Error procesando video: {str(e)}", exc_info=True)
            
            try:
                video_db = VideoDataset(
                    categoria_id=categoria_id,
                    usuario_id=usuario_id,
                    sena=sena.upper(),
                    ruta_video="error",
                    duracion_segundos=0,
                    fps=30.0,
                    resolucion=json.dumps({"ancho": 0, "alto": 0}),
                    tamaño_bytes=len(contenido_video) if 'contenido_video' in locals() else 0,
                    formato="webm",
                    frames_extraidos=0,
                    calidad_promedio=0.0,
                    procesado=False,
                    aprobado=False,
                    fecha_procesado=datetime.utcnow(),
                    notas=f"ERROR: {str(e)}"
                )
                db.add(video_db)
                db.commit()
                db.refresh(video_db)
                return video_db
            except Exception as inner_e:
                logger.error(f"✗ Error en fallback: {inner_e}")
                raise Exception(f"Error procesando video: {str(e)}")
            
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
            if temp_mp4_path and os.path.exists(temp_mp4_path):
                try:
                    os.unlink(temp_mp4_path)
                except:
                    pass
            gc.collect()

    def preparar_dataset_entrenamiento(
        self,
        db: Session,
        categoria_ids: Optional[List[int]] = None
    ) -> Dict[str, List[str]]:
        """
        NOTA: Este método ya no se usa directamente para entrenamiento
        porque los videos ya no viven en disco local. El nuevo flujo de
        entrenamiento descarga temporalmente desde Drive (ver drive_service
        y el servicio de entrenamiento actualizado).
        Se mantiene por compatibilidad con código que aún lo llame.
        """
        query = db.query(VideoDataset).filter(VideoDataset.aprobado == True)
        
        if categoria_ids:
            query = query.filter(VideoDataset.categoria_id.in_(categoria_ids))
        
        dataset = {}
        videos = query.all()
        
        for video in videos:
            sena = video.sena
            if sena not in dataset:
                dataset[sena] = []
            if video.ruta_video and os.path.exists(video.ruta_video):
                dataset[sena].append(video.ruta_video)
        
        logger.info(f"📊 Dataset preparado (solo locales): {len(dataset)} señas, {sum(len(v) for v in dataset.values())} videos")
        return dataset

    def obtener_estadisticas_dataset(self, db: Session) -> Dict[str, any]:
        try:
            total_videos = db.query(VideoDataset).count()
            videos_aprobados = db.query(VideoDataset).filter(VideoDataset.aprobado == True).count()
            total_frames = db.query(func.sum(VideoDataset.frames_extraidos)).scalar() or 0
            
            por_sena = db.query(
                VideoDataset.sena,
                func.count(VideoDataset.id).label('videos'),
                func.sum(VideoDataset.frames_extraidos).label('keypoints'),
                func.avg(VideoDataset.calidad_promedio).label('calidad_promedio')
            ).group_by(VideoDataset.sena).all()
            
            por_sena_list = []
            for item in por_sena:
                por_sena_list.append({
                    "sena": str(item[0]),
                    "videos": int(item[1]),
                    "frames": int(item[2]) if item[2] else 0,
                    "calidad_promedio": float(item[3]) if item[3] else 0.0
                })
            
            return {
                "total_videos": total_videos,
                "videos_aprobados": videos_aprobados,
                "total_frames": int(total_frames),
                "por_sena": sorted(por_sena_list, key=lambda x: x['videos'], reverse=True),
                "total_senas": len(por_sena_list),
                "tasa_aprobacion": round(videos_aprobados / total_videos * 100, 2) if total_videos > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"✗ Error obteniendo estadísticas: {str(e)}")
            return {
                "total_videos": 0,
                "videos_aprobados": 0,
                "total_frames": 0,
                "por_sena": [],
                "total_senas": 0,
                "tasa_aprobacion": 0
            }
    
    def aprobar_video(self, db: Session, video_id: int, aprobar: bool, notas: str = None) -> VideoDataset:
        try:
            video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
            
            if not video:
                raise Exception("Video no encontrado")
            
            video.aprobado = aprobar
            video.fecha_aprobado = datetime.utcnow() if aprobar else None
            
            if notas is not None and hasattr(video, 'notas'):
                video.notas = notas
            
            db.commit()
            db.refresh(video)
            
            logger.info(f"✓ Video {video_id} {'aprobado' if aprobar else 'rechazado'}")
            
            return video
            
        except Exception as e:
            db.rollback()
            logger.error(f"✗ Error aprobando video: {str(e)}")
            raise Exception(f"Error aprobando video: {str(e)}")

    def reprocesar_video(self, db: Session, video_id: int) -> VideoDataset:
        """
        NOTA: Solo funciona si el video aún existe localmente
        (es decir, si la subida a Drive falló en su momento).
        Si el video ya está solo en Drive, este método no puede
        reprocesarlo sin antes descargarlo (ver drive_service.descargar_archivo_de_drive).
        """
        try:
            video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()
            
            if not video:
                raise Exception("Video no encontrado")
            
            if not video.ruta_video or not os.path.exists(video.ruta_video):
                raise Exception(
                    "Archivo de video no encontrado localmente. "
                    "El video probablemente ya se subió a Drive y se eliminó localmente."
                )
            
            logger.info(f"🔄 Reprocesando video {video_id}: {video.sena}")
            
            keypoints_extraidos, calidad_promedio, fps_real, total_frames, duracion_real = self._procesar_video_directo(
                video.ruta_video, video.sena
            )
            
            video.frames_extraidos = keypoints_extraidos
            video.calidad_promedio = calidad_promedio
            video.fps = fps_real
            video.duracion_segundos = duracion_real
            video.procesado = True
            video.aprobado = keypoints_extraidos >= 3
            video.fecha_procesado = datetime.utcnow()
            video.notas = f"Reprocesado: {keypoints_extraidos} keypoints, Calidad: {calidad_promedio:.2f}"
            
            if video.aprobado:
                video.fecha_aprobado = datetime.utcnow()
            
            db.commit()
            db.refresh(video)
            
            logger.info(f"✓ Video {video_id} reprocesado: {keypoints_extraidos} keypoints")
            
            return video
            
        except Exception as e:
            db.rollback()
            logger.error(f"✗ Error reprocesando video: {str(e)}")
            raise Exception(f"Error reprocesando video: {str(e)}")

dataset_service = DatasetService()