import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def aumentar_imagenes_simple(carpeta, cantidad_objetivo=250):
    """Aumenta imágenes usando transformaciones simples con OpenCV"""
    imagenes = list(carpeta.glob("*.jpg")) + list(carpeta.glob("*.png")) + list(carpeta.glob("*.jpeg"))
    actual = len(imagenes)
    
    if actual >= cantidad_objetivo:
        print(f"✓ {carpeta.name}: Ya tiene {actual} imágenes")
        return
    
    faltantes = cantidad_objetivo - actual
    print(f"📸 {carpeta.name}: {actual} → generando {faltantes} más...")
    
    contador = actual
    while contador < cantidad_objetivo:
        # Imagen aleatoria
        img_path = np.random.choice(imagenes)
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
        
        # Aplicar transformaciones aleatorias
        # Rotación
        if np.random.rand() > 0.5:
            angle = np.random.randint(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        
        # Brillo
        if np.random.rand() > 0.5:
            value = np.random.randint(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=1, beta=value)
        
        # Flip horizontal
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        # Guardar
        nuevo_path = carpeta / f"aug_{contador:04d}.jpg"
        cv2.imwrite(str(nuevo_path), img)
        contador += 1
    
    print(f"   ✓ Completado: {contador} imágenes totales")

def balancear_dataset_completo():
    dataset_dir = Path("archivos_subidos/imagenes_entrenamiento")
    
    print("\n" + "="*60)
    print("BALANCEANDO DATASET CON DATA AUGMENTATION")
    print("="*60 + "\n")
    
    for carpeta in sorted(dataset_dir.iterdir()):
        if carpeta.is_dir():
            aumentar_imagenes_simple(carpeta, cantidad_objetivo=250)
    
    print("\n" + "="*60)
    print("¡Dataset balanceado exitosamente!")
    print("="*60 + "\n")

if __name__ == "__main__":
    balancear_dataset_completo()