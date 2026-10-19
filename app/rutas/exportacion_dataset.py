import csv
import hashlib
import io
import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from app.utilidades import archivos_firmados as firmas
from app.utilidades.base_datos import obtener_bd
from app.dependencias.permisos import requiere_permiso
from app.modelos.dataset import CategoriaDataset, VideoDataset
from app.modelos.usuario import Usuario
from app.servicios.drive_service import (
    descargar_videos_temporalmente,
    limpiar_carpeta_temporal,
)

router = APIRouter(prefix="/exportacion-dataset", tags=["exportacion_dataset"])

CAMPOS_MANIFIESTO = [
    "video_id", "sena", "categoria", "aprobado", "duracion_segundos",
    "fps", "frames", "fecha_subida", "archivo_en_zip", "sha256", "estado_archivo",
]


def _limpio(texto: Optional[str]) -> str:
    valor = re.sub(r"[^\w\-]+", "_", texto or "", flags=re.UNICODE).strip("_")
    return valor or "sin_nombre"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def _borrar(ruta: str) -> None:
    try:
        Path(ruta).unlink(missing_ok=True)
    except OSError:
        pass


def _ruta_local(video: VideoDataset) -> Optional[Path]:
    if not video.ruta_video:
        return None
    resuelta = firmas.resolver_ruta("videos_dataset", Path(video.ruta_video).name)
    return Path(resuelta) if resuelta is not None else None


def _iso(valor) -> Optional[str]:
    return valor.isoformat() if valor else None


@router.get("/zip")
def exportar_dataset_zip(
    categoria_id: Optional[int] = Query(None),
    sena: Optional[str] = Query(None, max_length=50),
    solo_aprobados: bool = Query(True),
    incluir_drive: bool = Query(False),
    usuario_actual: Usuario = Depends(requiere_permiso("dataset.videos.gestionar")),
    db: Session = Depends(obtener_bd),
):
    consulta = db.query(VideoDataset)
    if solo_aprobados:
        consulta = consulta.filter(VideoDataset.aprobado == True)
    if categoria_id is not None:
        consulta = consulta.filter(VideoDataset.categoria_id == categoria_id)
    if sena:
        consulta = consulta.filter(VideoDataset.sena == sena)

    videos: List[VideoDataset] = consulta.order_by(VideoDataset.sena, VideoDataset.id).all()
    if not videos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hay videos que coincidan con los filtros",
        )

    nombres_categoria: Dict[int, str] = {
        c.id: c.nombre for c in db.query(CategoriaDataset.id, CategoriaDataset.nombre).all()
    }

    locales: Dict[int, Path] = {}
    for video in videos:
        ruta = _ruta_local(video)
        if ruta is not None and ruta.is_file():
            locales[video.id] = ruta

    carpeta_temporal = f"temp_exportacion_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    desde_drive: Dict[int, Path] = {}
    usa_carpeta_temporal = False

    if incluir_drive:
        pendientes = [v for v in videos if v.id not in locales and v.drive_file_id]
        if pendientes:
            usa_carpeta_temporal = True
            try:
                rutas = descargar_videos_temporalmente(pendientes, carpeta_temporal=carpeta_temporal)
                desde_drive = {int(k): Path(v) for k, v in rutas.items() if v and Path(v).is_file()}
            except Exception:
                desde_drive = {}

    archivo_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    ruta_tmp = archivo_tmp.name
    archivo_tmp.close()

    filas: List[Dict[str, object]] = []
    incluidos = 0
    omitidos = 0
    por_sena: Dict[str, int] = {}

    try:
        with zipfile.ZipFile(ruta_tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for video in videos:
                ruta = locales.get(video.id) or desde_drive.get(video.id)
                categoria = nombres_categoria.get(video.categoria_id, f"categoria_{video.categoria_id}")
                fila: Dict[str, object] = {
                    "video_id": video.id,
                    "sena": video.sena,
                    "categoria": categoria,
                    "aprobado": video.aprobado,
                    "duracion_segundos": video.duracion_segundos,
                    "fps": getattr(video, "fps", None),
                    "frames": video.frames_extraidos,
                    "fecha_subida": _iso(getattr(video, "fecha_subida", None)),
                    "archivo_en_zip": "",
                    "sha256": "",
                    "estado_archivo": "",
                }

                if ruta is None:
                    fila["estado_archivo"] = "no_disponible"
                    omitidos += 1
                    filas.append(fila)
                    continue

                extension = ruta.suffix.lower() or ".mp4"
                nombre_zip = (
                    f"videos/{_limpio(categoria)}/{_limpio(video.sena)}/"
                    f"{_limpio(video.sena)}_{video.id}{extension}"
                )
                zf.write(ruta, arcname=nombre_zip, compress_type=zipfile.ZIP_STORED)

                fila["archivo_en_zip"] = nombre_zip
                fila["sha256"] = _sha256(ruta)
                fila["estado_archivo"] = "incluido"
                incluidos += 1
                por_sena[video.sena] = por_sena.get(video.sena, 0) + 1
                filas.append(fila)

            buffer = io.StringIO()
            escritor = csv.DictWriter(buffer, fieldnames=CAMPOS_MANIFIESTO)
            escritor.writeheader()
            escritor.writerows(filas)
            zf.writestr(
                "manifiesto.csv",
                buffer.getvalue().encode("utf-8-sig"),
                compress_type=zipfile.ZIP_DEFLATED,
            )

            resumen = {
                "generado": datetime.now().isoformat(),
                "filtros": {
                    "categoria_id": categoria_id,
                    "sena": sena,
                    "solo_aprobados": solo_aprobados,
                    "incluir_drive": incluir_drive,
                },
                "total_en_base_de_datos": len(videos),
                "incluidos": incluidos,
                "omitidos": omitidos,
                "videos_por_sena": dict(sorted(por_sena.items())),
            }
            zf.writestr(
                "resumen.json",
                json.dumps(resumen, ensure_ascii=False, indent=2).encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    except HTTPException:
        _borrar(ruta_tmp)
        raise
    except Exception as e:
        _borrar(ruta_tmp)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar el ZIP: {str(e)}",
        )
    finally:
        if usa_carpeta_temporal:
            try:
                limpiar_carpeta_temporal(carpeta_temporal)
            except Exception:
                pass

    if incluidos == 0:
        _borrar(ruta_tmp)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ningún video tiene su archivo disponible en el servidor",
        )

    nombre_descarga = f"dataset_signafree_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    return FileResponse(
        ruta_tmp,
        media_type="application/zip",
        filename=nombre_descarga,
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(_borrar, ruta_tmp),
    )