import random
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from app.utilidades.base_datos import SessionOwner
from app.modelos.dataset import VideoDataset
from app.modelos.modelo_adaptativo import ModeloAdaptativoSenas
from app.utilidades.normalizar_manos import normalizar_una_mano
from app.servicios.config_tipo_senas import detectar_tipo_sena

MODULO = "app.servicios.servicio_entrenamiento"
EPOCHS = 80
LR = 0.0008
BATCH = 8
NUM_FRAMES = 20

modulo = importlib.import_module(MODULO)
DatasetKeypointsAdaptativo = modulo.DatasetKeypointsAdaptativo
PerdidaContrastivaFocal = modulo.PerdidaContrastivaFocal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
carpeta = Path("archivos_subidos") / "secuencias_dataset"

db = SessionOwner()
try:
    filas = db.query(VideoDataset.id, VideoDataset.sena).filter(VideoDataset.aprobado == True).all()
finally:
    db.close()

secuencias, senas = [], []
for video_id, sena in filas:
    ruta = carpeta / f"{video_id}.npz"
    if not ruta.exists():
        continue
    datos = np.load(ruta)
    seq = normalizar_una_mano(datos["secuencia"].astype(np.float32))
    if seq.shape[0] != NUM_FRAMES:
        idx = np.linspace(0, seq.shape[0] - 1, NUM_FRAMES).round().astype(int)
        seq = seq[idx]
    secuencias.append(seq)
    senas.append(sena)

clases = sorted(set(senas))
clase_a_idx = {c: i for i, c in enumerate(clases)}
labels = [clase_a_idx[s] for s in senas]
tipos_senas = {c: detectar_tipo_sena(c) for c in clases}
print("Clases:", clases, "| tipos:", tipos_senas, "| total:", len(secuencias))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.arange(len(secuencias)), labels))

tr_seq = [secuencias[i] for i in train_idx]
va_seq = [secuencias[i] for i in val_idx]
tr_lab = [labels[i] for i in train_idx]
va_lab = [labels[i] for i in val_idx]
tr_sen = [senas[i] for i in train_idx]
va_sen = [senas[i] for i in val_idx]


def evaluar(model):
    val_ds = DatasetKeypointsAdaptativo(va_seq, va_lab, va_sen, NUM_FRAMES, False)
    loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    model.eval()
    with torch.no_grad():
        for kp, lb, tipos in loader:
            kp = kp.to(device)
            pred = model.forward_batch_mixto(kp, tipos).argmax(1).cpu().tolist()
            for r, p in zip(lb.tolist(), pred):
                matriz[r][p] += 1
    acc = np.trace(matriz) / matriz.sum()
    por_clase = {c: matriz[i][i] / max(1, matriz[i].sum()) for i, c in enumerate(clases)}
    return acc, por_clase, matriz


def correr(nombre, augmentation, perdida_focal):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_ds = DatasetKeypointsAdaptativo(tr_seq, tr_lab, tr_sen, NUM_FRAMES, augmentation)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    model = ModeloAdaptativoSenas(len(clases), tipos_senas, 12).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    if perdida_focal:
        criterio = PerdidaContrastivaFocal(clases, alpha=1.0, gamma=1.8, label_smoothing=0.015)
    else:
        criterio = nn.CrossEntropyLoss(label_smoothing=0.015)

    for epoch in range(EPOCHS):
        model.train()
        for kp, lb, tipos in loader:
            kp, lb = kp.to(device), lb.to(device)
            optimizer.zero_grad()
            salida = model.forward_batch_mixto(kp, tipos)
            loss = criterio(salida, lb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    acc, por_clase, matriz = evaluar(model)
    print(f"\n[{nombre}] aug={augmentation} focal={perdida_focal} | val_acc_final={acc * 100:.1f}%")
    print("  por clase:", {c: f"{a * 100:.0f}%" for c, a in por_clase.items()})
    print("  matriz:\n", matriz)


correr("A_base", augmentation=False, perdida_focal=False)
correr("B_solo_aug", augmentation=True, perdida_focal=False)
correr("C_solo_focal", augmentation=False, perdida_focal=True)
correr("D_completo", augmentation=True, perdida_focal=True)