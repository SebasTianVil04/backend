import shutil
from pathlib import Path

def copiar_dataset_completo():
    """Copia el dataset desde su ubicación original"""
    
    origen = Path("C:\Users\LENOVO\Desktop\Nueva carpeta (2)\SISTEMA\signafree_backend\modelo_ia\datos_entrenamiento")
    destino = Path("archivos_subidos/imagenes_entrenamiento")
    
    destino.mkdir(parents=True, exist_ok=True)
    
    print("Copiando dataset...")
    print(f"Origen: {origen}")
    print(f"Destino: {destino}")
    print("-" * 50)
    
    for carpeta in origen.iterdir():
        if carpeta.is_dir() and carpeta.name.startswith("letra_"):
            destino_carpeta = destino / carpeta.name
            
            if destino_carpeta.exists():
                print(f"⚠ {carpeta.name} ya existe, omitiendo...")
                continue
            
            shutil.copytree(carpeta, destino_carpeta)
            archivos = len(list(destino_carpeta.glob("*")))
            print(f"✓ {carpeta.name}: {archivos} archivos")
    
    print("-" * 50)
    print("¡Dataset copiado exitosamente!")

if __name__ == "__main__":
    copiar_dataset_completo()