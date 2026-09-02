# run.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def verificar_variables_entorno():
    """Verificar que las variables de entorno estén configuradas"""
    variables_requeridas = [
        'DATABASE_URL',
        'SECRET_KEY'
    ]
    
    variables_opcionales = {
        'APIPERU_TOKEN': 'API Peru no estará disponible (modo prueba)',
        'SMTP_HOST': 'El envío de emails no funcionará',
        'SMTP_USER': 'El envío de emails no funcionará'
    }
    
    faltantes = []
    advertencias = []
    
    # Verificar variables requeridas
    for var in variables_requeridas:
        if not os.getenv(var):
            faltantes.append(var)
    
    # Verificar variables opcionales
    for var, mensaje in variables_opcionales.items():
        if not os.getenv(var):
            advertencias.append(f"    {var}: {mensaje}")
    
    if faltantes:
        print("\n Faltan variables de entorno REQUERIDAS:")
        for var in faltantes:
            print(f"   - {var}")
        print("\n Copia .env.example a .env y configura las variables necesarias")
        sys.exit(1)
    
    print(" Variables de entorno requeridas configuradas")
    
    if advertencias:
        print("\n  Variables opcionales no configuradas:")
        for adv in advertencias:
            print(adv)


def crear_directorios():
    """Crear directorios necesarios para la aplicación"""
    directorios = [
        "archivos_subidos",
        "archivos_subidos/temp",
        "archivos_subidos/dataset_entrenamiento",
        "archivos_subidos/videos_lecciones",
        "archivos_subidos/imagenes_entrenamiento",
        "modelo_ia",
        "modelo_ia/datos_entrenamiento"
    ]
    
    for dir_path in directorios:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(" Directorios creados/verificados")


def verificar_dependencias():
    """Verificar que las dependencias críticas estén instaladas"""
    dependencias_criticas = {
        'fastapi': 'FastAPI',
        'sqlalchemy': 'SQLAlchemy',
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy'
    }
    
    faltantes = []
    
    for modulo, nombre in dependencias_criticas.items():
        try:
            __import__(modulo)
        except ImportError:
            faltantes.append(nombre)
    
    if faltantes:
        print("\n Faltan dependencias críticas:")
        for dep in faltantes:
            print(f"   - {dep}")
        print("\n Ejecuta: pip install -r requirements.txt")
        sys.exit(1)
    
    print(" Dependencias críticas verificadas")
    
    # Mostrar versiones de ML
    try:
        import torch
        print(f"   • PyTorch: {torch.__version__}")
    except:
        pass
    
    try:
        import mediapipe as mp
        print(f"   • MediaPipe: {mp.__version__}")
    except:
        pass




def mostrar_info_servidor(host, port, debug):
    """Mostrar información del servidor"""

    display_host = "localhost" if host == "0.0.0.0" else host
    
    print("                     SERVIDOR INICIADO")

    
    print(" URLs de acceso:")
    print(f"   • API Principal:    http://{display_host}:{port}")
    print(f"   • Documentación:    http://{display_host}:{port}/docs")
    print(f"   • ReDoc:            http://{display_host}:{port}/redoc")
    print(f"   • Health Check:     http://{display_host}:{port}/health")
    
    if host == "0.0.0.0":
        print(f"\n También disponible en:")
        print(f"   • http://127.0.0.1:{port}")
        print(f"   • http://127.0.0.1:{port}/docs")
    
    print(f"\n Modo: {'DEBUG (auto-reload)' if debug else 'PRODUCCIÓN'}")
    print("\n Presiona CTRL+C para detener el servidor")


def main():
    """Función principal"""
    try:

        
        # Verificaciones previas
        print(" Verificando configuración...\n")
        verificar_variables_entorno()
        crear_directorios()
        verificar_dependencias()
        
        # Importar configuración
        from app.utilidades.configuracion import configuracion
        
        # Mostrar información del servidor
        mostrar_info_servidor(
            configuracion.host,
            configuracion.port,
            configuracion.debug
        )
        
        # Iniciar servidor
        import uvicorn
        
        uvicorn.run(
            "app.main:app",
            host=configuracion.host,
            port=configuracion.port,
            reload=configuracion.debug,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"\nError al importar módulos: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n Error al ejecutar la aplicación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()