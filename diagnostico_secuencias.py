import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utilidades.base_datos import SessionLocal
from app.modelos.dataset import VideoDataset
from app.servicios.extractor_secuencias import VERSION_SECUENCIAS

SEC_DIR = Path("archivos_subidos") / "secuencias_dataset"

print(f"VERSION_SECUENCIAS actual: {VERSION_SECUENCIAS}")

db = SessionLocal()
try:
    videos = db.query(VideoDataset).filter(VideoDataset.aprobado == True).all()
    print(f"Videos aprobados: {len(videos)}")

    versiones = Counter()
    por_sena = {}
    faltantes = 0

    for v in videos:
        path = SEC_DIR / f"{v.id}.npz"
        if not path.exists():
            faltantes += 1
            continue
        data = np.load(path)
        ver = int(data.get("version", 0))
        versiones[ver] += 1
        if ver != VERSION_SECUENCIAS:
            continue
        por_sena.setdefault(v.sena, []).append(data["secuencia"])

    print(f"Versiones encontradas: {dict(versiones)}")
    print(f".npz faltantes: {faltantes}")
    print(f"Señas con .npz válidos: {len(por_sena)}")

    for sena, seqs in sorted(por_sena.items()):
        arr = np.stack(seqs)
        frames_c_mano = (np.abs(seqs).sum(axis=-1) > 0).sum(axis=-1)
        print(f"\n[{sena}] n={len(seqs)} shape={seqs[0].shape}")
        print(f"  media={arr.mean():.4f} std={arr.std():.4f}")
        print(f"  min={arr.min():.3f} max={arr.max():.3f}")
        print(f"  frames con mano: media={frames_c_mano.mean():.1f} min={frames_c_mano.min()}")

    if len(por_sena) >= 2:
        print("\n=== Distancia entre clases (L2 de medias) ===")
        medias = {s: np.stack(v).mean(axis=0).flatten() for s, v in por_sena.items()}
        senas = sorted(medias.keys())
        for i, s1 in enumerate(senas):
            for s2 in senas[i+1:]:
                d = float(np.linalg.norm(medias[s1] - medias[s2]))
                print(f"  {s1} vs {s2}: {d:.3f}")

        print("\n=== Varianza intra-clase (media) ===")
        for sena, seqs in sorted(por_sena.items()):
            arr = np.stack(seqs)
            print(f"  {sena}: {np.mean(np.std(arr, axis=0)):.4f}")
finally:
    db.close()