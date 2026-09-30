import os
import gc
import json
import uuid
import asyncio
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..modelos.dataset import CategoriaDataset, VideoDataset
from ..servicios.drive_service import (
    subir_archivo_a_drive,
    eliminar_archivo_de_drive,
    descargar_videos_temporalmente,
    limpiar_carpeta_temporal,
)
from ..servicios.extractor_secuencias import (
    extraer_secuencia_landmarks,
    secuencia_desde_keypoints,
    detectar_lateralidad,
    ResultadoSecuencia,
    VERSION_SECUENCIAS,
)

logger = logging.getLogger(__name__)


class DatasetService:

    def __init__(self):
        self.video_dir = Path("archivos_subidos") / "videos_dataset"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.secuencias_dir = Path("archivos_subidos") / "secuencias_dataset"
        self.secuencias_dir.mkdir(parents=True, exist_ok=True)

        self.num_frames = 20

        self.max_procesos_simultaneos = max(1, int(os.getenv("DATASET_MAX_PROCESOS", "2")))
        self._semaforo: Optional[asyncio.Semaphore] = None

        self.ffmpeg_disponible = self._verificar_ffmpeg()

        logger.info(f"DatasetService inicializado - FFmpeg: {'Disponible' if self.ffmpeg_disponible else 'No disponible'}")

    def _obtener_semaforo(self) -> asyncio.Semaphore:
        if self._semaforo is None:
            self._semaforo = asyncio.Semaphore(self.max_procesos_simultaneos)
        return self._semaforo

    def _verificar_ffmpeg(self) -> bool:
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _borrar_si_existe(ruta: Any) -> None:
        if not ruta:
            return
        try:
            if os.path.exists(ruta):
                os.remove(ruta)
        except Exception as e:
            logger.warning(f"No se pudo eliminar {ruta}: {e}")

    @staticmethod
    def _parsear_keypoints(keypoints: Optional[str]) -> Optional[list]:
        if not keypoints:
            return None
        try:
            datos_kp = json.loads(keypoints)
            if isinstance(datos_kp, list) and datos_kp:
                return datos_kp
        except Exception as e:
            logger.warning(f"No se pudo parsear 'keypoints' del cliente, se usará extracción por video: {e}")
        return None

    def _convertir_a_mp4(self, ruta_origen: str, ruta_mp4: str) -> bool:
        if not self.ffmpeg_disponible:
            logger.warning("FFmpeg no disponible, saltando conversión")
            return False

        try:
            cmd = [
                'ffmpeg',
                '-i', ruta_origen,
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-an',
                '-y',
                ruta_mp4
            ]

            logger.info(f"Convirtiendo a MP4: {os.path.basename(ruta_origen)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and os.path.exists(ruta_mp4):
                tamaño_mp4 = os.path.getsize(ruta_mp4)
                if tamaño_mp4 > 1024:
                    logger.info(f"Conversión exitosa: {tamaño_mp4:,} bytes")
                    return True
                logger.error(f"MP4 muy pequeño: {tamaño_mp4} bytes")
                self._borrar_si_existe(ruta_mp4)
                return False

            logger.error(f"FFmpeg falló con código {result.returncode}")
            if result.stderr:
                logger.error(f"Error FFmpeg: {result.stderr[:500]}")
            self._borrar_si_existe(ruta_mp4)
            return False

        except subprocess.TimeoutExpired:
            logger.error("Timeout en conversión FFmpeg (60s)")
            self._borrar_si_existe(ruta_mp4)
            return False
        except Exception as e:
            logger.error(f"Error en conversión: {e}")
            self._borrar_si_existe(ruta_mp4)
            return False

    def _procesar_secuencia(
        self,
        nombre_sena: str,
        ruta_video: Optional[str] = None,
        keypoints_cliente: Optional[list] = None,
    ) -> Tuple[int, float, float, int, float, Optional[ResultadoSecuencia]]:
        try:
            num_frames = self.num_frames

            if keypoints_cliente is not None and len(keypoints_cliente) > 0:
                arrs = [np.asarray(k, dtype=np.float32) for k in keypoints_cliente]
                resultado = secuencia_desde_keypoints(arrs, num_frames, fps=30.0)
            else:
                resultado = extraer_secuencia_landmarks(ruta_video, num_frames)

            if resultado is None:
                return 0, 0.5, 30.0, 0, 1.0, None

            duracion = resultado.frames_totales / resultado.fps if resultado.fps > 0 else 1.0

            return (
                resultado.keypoints_validos,
                resultado.calidad,
                resultado.fps,
                resultado.frames_totales,
                duracion,
                resultado
            )

        except Exception as e:
            logger.error(f"Error procesando secuencia: {e}", exc_info=True)
            return 0, 0.5, 30.0, 0, 1.0, None

    def _calcular_aprobacion(
        self,
        keypoints_extraidos: int,
        resultado: Optional[ResultadoSecuencia],
        nombre_sena: str,
    ) -> Tuple[bool, int]:
        umbral_densidad = max(3, int(self.num_frames * 0.6))

        if resultado is None:
            return False, umbral_densidad

        fps = resultado.fps if resultado.fps and resultado.fps > 0 else 30.0
        frames_detectados_minimos = max(5, int(fps * 0.3))

        aprobado = (
            keypoints_extraidos >= umbral_densidad
            and resultado.frames_detectados >= frames_detectados_minimos
        )
        return aprobado, umbral_densidad

    def _guardar_secuencia(self, video_id: int, resultado: Optional[ResultadoSecuencia]) -> None:
        if resultado is None:
            return

        try:
            np.savez_compressed(
                self.secuencias_dir / f"{video_id}.npz",
                secuencia=resultado.secuencia,
                estado=resultado.estado,
                version=np.array(VERSION_SECUENCIAS),
            )
            logger.info(f"Secuencia guardada: {video_id}.npz {resultado.secuencia.shape}")
        except Exception as e:
            logger.warning(f"No se pudo guardar la secuencia del video {video_id}: {e}")

    def _guardar_keypoints_cliente(self, video_id: int, keypoints_cliente: Optional[list]) -> None:
        if not keypoints_cliente:
            return

        try:
            arr = np.array(keypoints_cliente, dtype=np.float32)
            np.savez_compressed(self.secuencias_dir / f"{video_id}_kp_cliente.npz", keypoints=arr)
            logger.info(f"Keypoints del cliente guardados: {video_id}_kp_cliente.npz {arr.shape}")
        except Exception as e:
            logger.warning(f"No se pudieron guardar los keypoints del cliente del video {video_id}: {e}")

    def _cargar_keypoints_cliente(self, video_id: int) -> Optional[list]:
        ruta = self.secuencias_dir / f"{video_id}_kp_cliente.npz"
        if not ruta.exists():
            return None

        try:
            with np.load(ruta) as datos:
                return list(datos["keypoints"])
        except Exception as e:
            logger.warning(f"No se pudieron leer los keypoints del cliente del video {video_id}: {e}")
            return None

    def _subir_video_a_drive_y_limpiar(self, ruta_video_local: str, formato: str) -> Dict[str, Optional[str]]:
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

    def _limpiar_tras_error(self, rutas: List[Any], drive_file_id: Optional[str]) -> None:
        for ruta in rutas:
            self._borrar_si_existe(ruta)

        if drive_file_id:
            try:
                eliminar_archivo_de_drive(drive_file_id)
            except Exception as e:
                logger.warning(f"[Drive] No se pudo eliminar el archivo huérfano {drive_file_id}: {e}")

    async def subir_video_dataset(
        self,
        db: Session,
        archivo: Any,
        categoria_id: int,
        sena: str,
        usuario_id: int,
        keypoints: Optional[str] = None,
    ) -> VideoDataset:
        if hasattr(archivo, 'read'):
            contenido_video = await archivo.read()
        else:
            contenido_video = archivo

        nombre_archivo = getattr(archivo, 'filename', '') or ''

        async with self._obtener_semaforo():
            return await run_in_threadpool(
                self._procesar_y_guardar_video,
                db,
                contenido_video,
                nombre_archivo,
                categoria_id,
                sena,
                usuario_id,
                keypoints,
            )

    def _procesar_y_guardar_video(
        self,
        db: Session,
        contenido_video: bytes,
        nombre_archivo: str,
        categoria_id: int,
        sena: str,
        usuario_id: int,
        keypoints: Optional[str],
    ) -> VideoDataset:
        ruta_original: Optional[Path] = None
        ruta_conversion: Optional[Path] = None
        ruta_mp4_final: Optional[Path] = None
        drive_file_id: Optional[str] = None
        video_guardado = False

        try:
            if len(contenido_video) < 1024:
                raise ValueError("Archivo de video demasiado pequeño")

            logger.info(f"Procesando video: {sena}, {len(contenido_video):,} bytes")

            keypoints_cliente = self._parsear_keypoints(keypoints)

            extension_original = Path(nombre_archivo).suffix.lower()
            if extension_original not in ('.webm', '.mp4'):
                extension_original = '.webm'

            base = uuid.uuid4().hex

            ruta_original = self.video_dir / f"{base}{extension_original}"
            ruta_original.write_bytes(contenido_video)

            logger.info(f"Video original guardado: {ruta_original}")

            formato_guardado = extension_original.lstrip('.')
            ruta_video_final = ruta_original

            if self.ffmpeg_disponible:
                ruta_mp4_final = self.video_dir / f"{base}.mp4"
                ruta_conversion = self.video_dir / f"{base}_conv.mp4"

                if self._convertir_a_mp4(str(ruta_original), str(ruta_conversion)):
                    os.replace(ruta_conversion, ruta_mp4_final)
                    if ruta_original != ruta_mp4_final:
                        self._borrar_si_existe(ruta_original)
                    formato_guardado = "mp4"
                    ruta_video_final = ruta_mp4_final
                    logger.info(f"MP4 guardado exitosamente: {ruta_mp4_final}")
                else:
                    logger.warning("Conversión a MP4 falló, se usa el video original")
            else:
                logger.warning("FFmpeg no disponible, se procesa el video original directamente")

            logger.info(
                "Extrayendo secuencia "
                + ("(keypoints del cliente)" if keypoints_cliente else f"de {formato_guardado.upper()}")
            )
            (keypoints_extraidos, calidad_promedio, fps_real,
             total_frames, duracion_real, secuencia) = self._procesar_secuencia(
                sena, str(ruta_video_final), keypoints_cliente
            )

            mano_detectada = detectar_lateralidad(secuencia) if secuencia is not None else "no_detectada"

            aprobado, _ = self._calcular_aprobacion(keypoints_extraidos, secuencia, sena)

            drive_info = self._subir_video_a_drive_y_limpiar(str(ruta_video_final), formato_guardado)
            drive_file_id = drive_info["drive_file_id"]
            subio_a_drive = drive_file_id is not None

            ruta_para_bd = None if subio_a_drive else str(ruta_video_final)

            notas_extra = ""
            if not subio_a_drive:
                notas_extra = " | ADVERTENCIA: no se pudo subir a Drive, archivo conservado localmente"

            video_db = VideoDataset(
                categoria_id=categoria_id,
                usuario_id=usuario_id,
                sena=sena.upper(),
                ruta_video=ruta_para_bd,
                drive_file_id=drive_file_id,
                drive_url=drive_info["drive_url"],
                duracion_segundos=duracion_real,
                fps=fps_real,
                resolucion=json.dumps({"ancho": 640, "alto": 480}),
                tamaño_bytes=len(contenido_video),
                formato=formato_guardado,
                mano_detectada=mano_detectada,
                frames_extraidos=keypoints_extraidos,
                calidad_promedio=calidad_promedio,
                procesado=True,
                aprobado=aprobado,
                fecha_procesado=datetime.utcnow(),
                notas=f"Formato: {formato_guardado.upper()}, Keypoints: {keypoints_extraidos}/{self.num_frames}, Calidad: {calidad_promedio:.2f}, Mano: {mano_detectada}{notas_extra}"
            )

            if aprobado:
                video_db.fecha_aprobado = datetime.utcnow()

            db.add(video_db)
            db.commit()
            video_guardado = True
            db.refresh(video_db)

            self._guardar_secuencia(video_db.id, secuencia)
            self._guardar_keypoints_cliente(video_db.id, keypoints_cliente)

            estado = "APROBADO" if aprobado else "PENDIENTE"
            drive_estado = "Drive OK" if subio_a_drive else "Solo local"
            logger.info(f"{estado} - Video {video_db.id} - {keypoints_extraidos} keypoints - {formato_guardado.upper()} - Mano: {mano_detectada} - {drive_estado}")

            return video_db

        except Exception as e:
            db.rollback()
            logger.error(f"Error procesando video: {str(e)}", exc_info=True)

            if not video_guardado:
                self._limpiar_tras_error(
                    [ruta_original, ruta_conversion, ruta_mp4_final],
                    drive_file_id,
                )
            raise

        finally:
            gc.collect()

    def cargar_secuencias_entrenamiento(
        self,
        db: Session,
        categoria_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        query = db.query(VideoDataset).filter(VideoDataset.aprobado == True)

        if categoria_ids:
            query = query.filter(VideoDataset.categoria_id.in_(categoria_ids))

        X: List[np.ndarray] = []
        y: List[str] = []

        for video in query.all():
            ruta = self.secuencias_dir / f"{video.id}.npz"
            if not ruta.exists():
                continue

            with np.load(ruta) as datos:
                if "version" not in datos or int(datos["version"]) != VERSION_SECUENCIAS:
                    continue
                if not (datos["estado"] > 0).any():
                    continue
                secuencia = datos["secuencia"]

            if secuencia.shape[0] != self.num_frames:
                logger.warning(
                    f"Secuencia del video {video.id} tiene {secuencia.shape[0]} frames, "
                    f"se esperaban {self.num_frames}. Video descartado (reprocesar_video corrige esto)."
                )
                continue

            X.append(secuencia)
            y.append(video.sena)

        logger.info(f"Secuencias cargadas: {len(X)} de {len(set(y))} señas")

        if not X:
            return np.zeros((0, self.num_frames, 126), dtype=np.float32), []

        return np.array(X, dtype=np.float32), y

    def preparar_dataset_entrenamiento(
        self,
        db: Session,
        categoria_ids: Optional[List[int]] = None
    ) -> Dict[str, List[str]]:
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

        logger.info(f"Dataset preparado (solo locales): {len(dataset)} señas, {sum(len(v) for v in dataset.values())} videos")
        return dataset

    def obtener_estadisticas_dataset(self, db: Session) -> Dict[str, Any]:
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
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
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

            if aprobar:
                video.rechazado = False
                video.fecha_rechazado = None

            if notas is not None and hasattr(video, 'notas'):
                video.notas = notas

            db.commit()
            db.refresh(video)

            logger.info(f"Video {video_id} {'aprobado' if aprobar else 'rechazado'}")

            return video

        except Exception as e:
            db.rollback()
            logger.error(f"Error aprobando video: {str(e)}")
            raise Exception(f"Error aprobando video: {str(e)}")

    def reprocesar_video(self, db: Session, video_id: int) -> VideoDataset:
        try:
            video = db.query(VideoDataset).filter(VideoDataset.id == video_id).first()

            if not video:
                raise Exception("Video no encontrado")

            keypoints_cliente = self._cargar_keypoints_cliente(video_id)

            if keypoints_cliente is None and (not video.ruta_video or not os.path.exists(video.ruta_video)):
                raise Exception(
                    "Archivo de video no encontrado localmente y no hay keypoints del cliente guardados. "
                    "El video probablemente ya se subió a Drive y se eliminó localmente."
                )

            logger.info(f"Reprocesando video {video_id}: {video.sena}")

            (keypoints_extraidos, calidad_promedio, fps_real,
             total_frames, duracion_real, secuencia) = self._procesar_secuencia(
                video.sena, video.ruta_video, keypoints_cliente
            )

            video.mano_detectada = detectar_lateralidad(secuencia) if secuencia is not None else "no_detectada"

            aprobado, _ = self._calcular_aprobacion(keypoints_extraidos, secuencia, video.sena)

            video.frames_extraidos = keypoints_extraidos
            video.calidad_promedio = calidad_promedio
            video.fps = fps_real
            video.duracion_segundos = duracion_real
            video.procesado = True
            video.aprobado = aprobado
            video.fecha_procesado = datetime.utcnow()
            video.notas = f"Reprocesado: {keypoints_extraidos}/{self.num_frames} keypoints, Calidad: {calidad_promedio:.2f}, Mano: {video.mano_detectada}"

            if video.aprobado:
                video.fecha_aprobado = datetime.utcnow()

            db.commit()
            db.refresh(video)

            self._guardar_secuencia(video.id, secuencia)

            logger.info(f"Video {video_id} reprocesado: {keypoints_extraidos} keypoints")

            return video

        except Exception as e:
            db.rollback()
            logger.error(f"Error reprocesando video: {str(e)}")
            raise Exception(f"Error reprocesando video: {str(e)}")

    def reprocesar_todos(
        self,
        db: Session,
        categoria_ids: Optional[List[int]] = None,
        incluir_no_aprobados: bool = True,
        incluir_aprobados: bool = True,
    ) -> Dict[str, Any]:
        query = db.query(VideoDataset)
        if categoria_ids:
            query = query.filter(VideoDataset.categoria_id.in_(categoria_ids))

        estados_incluidos = []
        if incluir_aprobados:
            estados_incluidos.append(True)
        if incluir_no_aprobados:
            estados_incluidos.append(False)
        if estados_incluidos:
            query = query.filter(VideoDataset.aprobado.in_(estados_incluidos))

        videos = query.all()

        ids_con_kp = {
            v.id for v in videos
            if (self.secuencias_dir / f"{v.id}_kp_cliente.npz").exists()
        }
        ids_locales = {
            v.id for v in videos
            if v.id not in ids_con_kp and v.ruta_video and os.path.exists(v.ruta_video)
        }
        solo_drive = [
            v for v in videos
            if v.id not in ids_con_kp and v.id not in ids_locales and v.drive_file_id
        ]
        ids_drive = {v.id for v in solo_drive}
        sin_archivo = [
            v for v in videos
            if v.id not in ids_con_kp and v.id not in ids_locales and v.id not in ids_drive
        ]
        ids_sin_archivo = {v.id for v in sin_archivo}

        logger.info(
            f"Reprocesamiento masivo: {len(videos)} videos totales | "
            f"{len(ids_con_kp)} con keypoints de cliente | {len(ids_locales)} locales | "
            f"{len(solo_drive)} solo en Drive | {len(sin_archivo)} sin archivo disponible"
        )

        rutas_temporales: Dict[int, str] = {}
        carpeta_temporal = f"temp_reproceso_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if solo_drive:
            logger.info(f"Descargando {len(solo_drive)} videos desde Drive para reprocesar...")
            rutas_temporales = descargar_videos_temporalmente(
                solo_drive, carpeta_temporal=carpeta_temporal
            )

        resumen = {
            "total": len(videos),
            "reprocesados_ok": 0,
            "pasaron_a_aprobado": 0,
            "pasaron_a_no_aprobado": 0,
            "sin_cambio": 0,
            "fallidos": [],
            "sin_archivo_disponible": [v.id for v in sin_archivo],
        }

        try:
            for video in videos:
                if video.id in ids_sin_archivo:
                    continue

                keypoints_cliente = (
                    self._cargar_keypoints_cliente(video.id) if video.id in ids_con_kp else None
                )
                ruta = None

                if keypoints_cliente is None:
                    if video.ruta_video and os.path.exists(video.ruta_video):
                        ruta = video.ruta_video
                    else:
                        ruta = rutas_temporales.get(video.id)

                    if not ruta or not os.path.exists(ruta):
                        resumen["fallidos"].append({"id": video.id, "motivo": "archivo no disponible"})
                        continue

                try:
                    aprobado_antes = video.aprobado

                    (keypoints_extraidos, calidad_promedio, fps_real,
                     total_frames, duracion_real, secuencia) = self._procesar_secuencia(
                        video.sena, ruta, keypoints_cliente
                    )

                    video.mano_detectada = detectar_lateralidad(secuencia) if secuencia is not None else "no_detectada"

                    aprobado_nuevo, _ = self._calcular_aprobacion(keypoints_extraidos, secuencia, video.sena)

                    video.frames_extraidos = keypoints_extraidos
                    video.calidad_promedio = calidad_promedio
                    video.fps = fps_real
                    video.duracion_segundos = duracion_real
                    video.procesado = True
                    video.aprobado = aprobado_nuevo
                    video.fecha_procesado = datetime.utcnow()
                    video.notas = (
                        f"Reprocesado (lote): {keypoints_extraidos}/{self.num_frames} "
                        f"keypoints, Calidad: {calidad_promedio:.2f}, Mano: {video.mano_detectada}"
                    )

                    if video.aprobado:
                        video.fecha_aprobado = datetime.utcnow()
                    else:
                        video.fecha_aprobado = None

                    db.commit()
                    db.refresh(video)

                    self._guardar_secuencia(video.id, secuencia)

                    resumen["reprocesados_ok"] += 1
                    if aprobado_antes == video.aprobado:
                        resumen["sin_cambio"] += 1
                    elif video.aprobado:
                        resumen["pasaron_a_aprobado"] += 1
                    else:
                        resumen["pasaron_a_no_aprobado"] += 1

                    logger.info(
                        f"Video {video.id} reprocesado: {keypoints_extraidos} keypoints, "
                        f"aprobado {aprobado_antes} -> {video.aprobado}"
                    )
                except Exception as e:
                    db.rollback()
                    logger.error(f"Error reprocesando video {video.id}: {e}")
                    resumen["fallidos"].append({"id": video.id, "motivo": str(e)})
        finally:
            if solo_drive:
                try:
                    limpiar_carpeta_temporal(carpeta_temporal)
                except Exception as e:
                    logger.warning(f"No se pudo limpiar carpeta temporal '{carpeta_temporal}': {e}")
            gc.collect()

        logger.info(f"Reprocesamiento masivo terminado: {resumen}")
        return resumen


dataset_service = DatasetService()