import numpy as np
from pathlib import Path

from app.utilidades.base_datos import SessionOwner
from app.modelos.dataset import VideoDataset
from app.utilidades.normalizar_manos import normalizar_una_mano

carpeta = Path("archivos_subidos") / "secuencias_dataset"

db = SessionOwner()
try:
    filas = db.query(VideoDataset.id, VideoDataset.sena).filter(VideoDataset.aprobado == True).all()
finally:
    db.close()

ids, etiquetas, caracteristicas = [], [], []

for video_id, sena in filas:
    ruta = carpeta / f"{video_id}.npz"
    if not ruta.exists():
        continue
    seq = normalizar_una_mano(np.load(ruta)["secuencia"])
    mano = seq[:, :63]
    ids.append(video_id)
    etiquetas.append(sena)
    caracteristicas.append(np.concatenate([mano.mean(axis=0), mano.std(axis=0)]))

if not ids:
    print("No se encontraron secuencias guardadas")
    raise SystemExit

X = np.array(caracteristicas, dtype=np.float32)
y = np.array(etiquetas)
clases = sorted(set(etiquetas))
indice = {c: i for i, c in enumerate(clases)}

dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(dist, np.inf)
vecino = dist.argmin(axis=1)
pred = y[vecino]

matriz = np.zeros((len(clases), len(clases)), dtype=int)
errores = []
for i in range(len(y)):
    matriz[indice[y[i]]][indice[pred[i]]] += 1
    if y[i] != pred[i]:
        errores.append((ids[i], y[i], pred[i], ids[vecino[i]]))

print("Accuracy 1-NN leave-one-out:", f"{(pred == y).mean() * 100:.1f}%")
print()
print("Matriz de confusion (filas=real, columnas=predicho)")
print("      " + " ".join(f"{c:>5}" for c in clases))
for c in clases:
    print(f"{c:>5} " + " ".join(f"{matriz[indice[c]][indice[o]]:>5}" for o in clases))

print()
print("Accuracy por clase")
for c in clases:
    total = matriz[indice[c]].sum()
    print(f"{c}: {matriz[indice[c]][indice[c]] / total * 100:.1f}%")

centroides = {c: X[y == c].mean(axis=0) for c in clases}
print()
print("Distancia entre centroides")
print("      " + " ".join(f"{c:>6}" for c in clases))
for a in clases:
    print(f"{a:>5} " + " ".join(f"{np.linalg.norm(centroides[a] - centroides[b]):>6.2f}" for b in clases))

print()
print("Primeros 15 errores (id real, real, predicho, id del vecino)")
for e in errores[:15]:
    print(e)