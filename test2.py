import numpy as np
from pathlib import Path
from collections import defaultdict

from app.utilidades.base_datos import SessionOwner
from app.modelos.dataset import VideoDataset

carpeta = Path("archivos_subidos") / "secuencias_dataset"

db = SessionOwner()
try:
    filas = db.query(VideoDataset.id, VideoDataset.sena).filter(VideoDataset.aprobado == True).all()
finally:
    db.close()

resumen = defaultdict(lambda: {"videos": 0, "sin_mano": 0, "slot0": 0, "slot1": 0, "frames": []})

for video_id, sena in filas:
    ruta = carpeta / f"{video_id}.npz"
    if not ruta.exists():
        continue
    datos = np.load(ruta)
    secuencia = datos["secuencia"]
    estado = datos["estado"]
    r = resumen[sena]
    r["videos"] += 1
    r["frames"].append(int((estado > 0).sum()))
    if not (estado > 0).any():
        r["sin_mano"] += 1
    if np.any(secuencia[:, :63] != 0):
        r["slot0"] += 1
    if np.any(secuencia[:, 63:] != 0):
        r["slot1"] += 1

if not resumen:
    print("No se encontraron secuencias guardadas")

for sena, r in sorted(resumen.items()):
    frames = r["frames"]
    print(
        f"{sena}: videos={r['videos']} sin_mano={r['sin_mano']} "
        f"slot0={r['slot0']} slot1={r['slot1']} "
        f"frames_con_mano min/media={min(frames)}/{sum(frames) / len(frames):.1f}"
    )