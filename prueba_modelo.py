import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import contextlib
import numpy as np
from pathlib import Path


@contextlib.contextmanager
def silenciar_stderr_nativo():
    """
    Los mensajes 'W0000 ... inference_feedback_manager' los escribe Abseil/TFLite
    directo al file descriptor 2 (stderr) del sistema operativo, sin pasar por el
    logging de Python. Por eso ni GLOG_minloglevel ni warnings.filterwarnings los
    detienen del todo: hay que redirigir el file descriptor mismo mientras se
    ejecuta el código que los dispara (carga e inferencia de mediapipe/TFLite).
    """
    stderr_fd = sys.stderr.fileno()
    copia_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(copia_stderr, stderr_fd)
        os.close(devnull)
        os.close(copia_stderr)

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cv2
from app.utilidades.base_datos import SessionLocal
from app.modelos.entrenamiento import ModeloIA
from app.modelos.dataset import VideoDataset
from app.servicios.servicio_reconocimiento import ServicioReconocimiento

print("=" * 60)
print("PRUEBA 1: ¿Qué modelo está activo en BD?")
print("=" * 60)

db = SessionLocal()
try:
    activos = db.query(ModeloIA).filter(ModeloIA.activo == True).all()
    for m in activos:
        print(f"  id={m.id} nombre={m.nombre}")
        print(f"  ruta={m.ruta_archivo}")
        print(f"  accuracy={m.accuracy}")
        print(f"  fecha={m.fecha_entrenamiento}")

    if not activos:
        print("  ❌ NO HAY MODELO ACTIVO")
        sys.exit(1)

    modelo_activo = activos[0]
finally:
    db.close()

print()
print("=" * 60)
print("PRUEBA 2: ¿El .pth cargado coincide con el activo?")
print("=" * 60)

with silenciar_stderr_nativo():
    reconocedor = ServicioReconocimiento(ruta_modelo=modelo_activo.ruta_archivo)
print(f"  Clases cargadas: {reconocedor.clases}")
print(f"  num_frames: {reconocedor.num_frames}")
print(f"  accuracy: {reconocedor.accuracy}")
print(f"  extractor_version: {reconocedor.extractor_version}")

# Verificar arquitectura
if hasattr(reconocedor.model, 'hidden_size'):
    print(f"  hidden_size: {reconocedor.model.hidden_size}")
if hasattr(reconocedor.model, 'num_layers'):
    print(f"  num_layers: {reconocedor.model.num_layers}")

n_params = sum(p.numel() for p in reconocedor.model.parameters())
print(f"  Parámetros totales: {n_params:,}")

print()
print("=" * 60)
print("PRUEBA 3: Predicción sobre 3 videos del propio dataset")
print("=" * 60)

db = SessionLocal()
try:
    for sena_test in reconocedor.clases:
        videos = db.query(VideoDataset).filter(
            VideoDataset.sena == sena_test,
            VideoDataset.aprobado == True,
        ).limit(3).all()

        for v in videos:
            if not v.ruta_video or not Path(v.ruta_video).exists():
                continue
            try:
                with silenciar_stderr_nativo():
                    sena_pred, conf, detalles = reconocedor.predecir_desde_video(v.ruta_video, sena_test)
                top3 = sorted(detalles.get("alternativas", []), key=lambda x: -x["confianza"])[:3]
                top3_str = " | ".join([f"{a['sena']}:{a['confianza']*100:.0f}%" for a in top3])
                estado = "✅" if sena_pred == sena_test else "❌"
                print(f"  {estado} esperado={sena_test} pred={sena_pred} ({conf*100:.1f}%) [{top3_str}]")
            except Exception as e:
                print(f"  ⚠ Video {v.id} error: {e}")
finally:
    db.close()

print()
print("=" * 60)
print("PRUEBA 4: Predicción sobre un frame de webcam (si existe)")
print("=" * 60)

# Buscar la última secuencia del cliente para probar como si fuera webcam
sec_dir = Path("archivos_subidos") / "secuencias_dataset"
kp_cliente = list(sec_dir.glob("*_kp_cliente.npz"))
if kp_cliente:
    print(f"  Encontrados {len(kp_cliente)} archivos de keypoints del cliente")
    for kp_file in kp_cliente[:3]:
        try:
            data = np.load(kp_file)
            kps = list(data["keypoints"])
            with silenciar_stderr_nativo():
                sena_pred, conf, detalles = reconocedor.predecir_desde_keypoints(kps)
            top3 = sorted(detalles.get("alternativas", []), key=lambda x: -x["confianza"])[:3]
            top3_str = " | ".join([f"{a['sena']}:{a['confianza']*100:.0f}%" for a in top3])
            print(f"  {kp_file.name}: pred={sena_pred} ({conf*100:.1f}%) [{top3_str}]")
        except Exception as e:
            print(f"  ⚠ Error con {kp_file.name}: {e}")
else:
    print("  No hay keypoints del cliente guardados")