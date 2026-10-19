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

grupos = defaultdict(lambda: defaultdict(list))

for video_id, sena in filas:
    ruta = carpeta / f"{video_id}.npz"
    if not ruta.exists():
        continue
    secuencia = np.load(ruta)["secuencia"]
    s0 = np.any(secuencia[:, :63] != 0, axis=1)
    s1 = np.any(secuencia[:, 63:] != 0, axis=1)
    if s0.any() and s1.any():
        tipo = "ambas"
        ref = secuencia[s0][:, :63].reshape(-1, 21, 3)
    elif s0.any():
        tipo = "solo_slot0"
        ref = secuencia[s0][:, :63].reshape(-1, 21, 3)
    else:
        tipo = "solo_slot1"
        ref = secuencia[s1][:, 63:].reshape(-1, 21, 3)
    grupos[sena][tipo].append({
        "id": video_id,
        "muneca": ref[:, 0, :].mean(axis=0),
        "x_min": float(ref[:, :, 0].min()),
        "x_max": float(ref[:, :, 0].max()),
        "frames_s0": int(s0.sum()),
        "frames_s1": int(s1.sum()),
    })

for sena in sorted(grupos):
    print(f"\n=== {sena} ===")
    for tipo, items in sorted(grupos[sena].items()):
        muneca = np.mean([i["muneca"] for i in items], axis=0)
        x_min = min(i["x_min"] for i in items)
        x_max = max(i["x_max"] for i in items)
        f0 = np.mean([i["frames_s0"] for i in items])
        f1 = np.mean([i["frames_s1"] for i in items])
        ids = [i["id"] for i in items][:5]
        print(
            f"{tipo}: n={len(items)} ids={ids} "
            f"muneca_media=({muneca[0]:.3f}, {muneca[1]:.3f}, {muneca[2]:.3f}) "
            f"x_rango=({x_min:.3f}, {x_max:.3f}) "
            f"frames_slot0={f0:.1f} frames_slot1={f1:.1f}"
        )