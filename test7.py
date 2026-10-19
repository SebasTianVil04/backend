import logging
import importlib

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.utilidades.base_datos import SessionOwner
from app.modelos.dataset import VideoDataset

MODULO = "app.servicios.servicio_entrenamiento"
EPOCHS = 250

modulo = importlib.import_module(MODULO)
servicio = modulo.servicio_entrenamiento

db = SessionOwner()
try:
    categorias = sorted({
        c for (c,) in db.query(VideoDataset.categoria_id)
        .filter(VideoDataset.aprobado == True).distinct().all()
        if c is not None
    })
finally:
    db.close()

print("Categorias:", categorias)

nombre = "modelo_diagnostico_test7"
try:
    servicio.entrenar(nombre, categorias, EPOCHS, desactivar_otros=False)
except Exception as e:
    print("ERROR:", e)

p = servicio.obtener_progreso_entrenamiento(nombre)
print("\nEstado:", p["estado"], "| epoca:", p["epoch_actual"], "/", p["total_epochs"])
print("Accuracy:", p["accuracy"], "| train_acc:", p["train_accuracy"])
print("Por clase:", p["accuracy_por_clase"])
print("Advertencias:", p["advertencias_dataset"])