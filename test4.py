import numpy as np
from pathlib import Path
from collections import defaultdict

from app.utilidades.base_datos import SessionOwner
from app.modelos.dataset import VideoDataset
from app.utilidades.normalizar_manos import normalizar_una_mano

carpeta = Path("archivos_subidos") / "secuencias_dataset"

db = SessionOwner()
try:
    filas = db.query(VideoDataset.id, VideoDataset.sena).filter(VideoDataset.aprobado == True).all()
finally:
    db.close()

resumen = defaultdict(lambda: {"n": 0, "slot1": 0, "x_min": [], "x_max": []})

for video_id, sena in filas:
    ruta = carpeta / f"{video_id}.npz"
    if not ruta.exists():
        continue
    seq = normalizar_una_mano(np.load(ruta)["secuencia"])
    r = resumen[sena]
    r["n"] += 1
    if np.any(seq[:, 63:] != 0):
        r["slot1"] += 1
    mano = seq[:, :63].reshape(-1, 21, 3)
    r["x_min"].append(float(mano[:, :, 0].min()))
    r["x_max"].append(float(mano[:, :, 0].max()))

if not resumen:
    print("No se encontraron secuencias guardadas")

for sena, r in sorted(resumen.items()):
    print(
        f"{sena}: videos={r['n']} con_slot1={r['slot1']} "
        f"x_min_medio={np.mean(r['x_min']):.3f} x_max_medio={np.mean(r['x_max']):.3f}"
    )