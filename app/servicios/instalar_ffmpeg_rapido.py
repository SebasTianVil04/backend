import os
import sys
import urllib.request
import zipfile
import tempfile
import shutil

def descargar_ffmpeg_portable():
    """Descarga FFmpeg portable para Windows."""
    print("🚀 Descargando FFmpeg portable...")
    
    # URL de FFmpeg portable para Windows
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "ffmpeg.zip")
    extract_dir = os.path.join(temp_dir, "ffmpeg_extract")
    
    try:
        # Descargar
        print("📥 Descargando...")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extraer
        print("📦 Extrayendo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Encontrar ffmpeg.exe
        ffmpeg_exe = None
        for root, dirs, files in os.walk(extract_dir):
            if 'ffmpeg.exe' in files:
                ffmpeg_exe = os.path.join(root, 'ffmpeg.exe')
                break
        
        if ffmpeg_exe:
            # Crear directorio en el proyecto
            project_dir = os.path.dirname(os.path.abspath(__file__))
            ffmpeg_dir = os.path.join(project_dir, "ffmpeg")
            os.makedirs(ffmpeg_dir, exist_ok=True)
            
            # Copiar ffmpeg.exe
            shutil.copy2(ffmpeg_exe, os.path.join(ffmpeg_dir, "ffmpeg.exe"))
            
            print(f"✅ FFmpeg instalado en: {ffmpeg_dir}")
            print("🔧 Agregando al PATH...")
            
            # Agregar al PATH del sistema
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ['PATH']
            
            # Verificar instalación
            result = os.system('ffmpeg -version')
            if result == 0:
                print("🎉 FFmpeg funciona correctamente!")
            else:
                print("⚠️ FFmpeg instalado pero no funciona automáticamente")
                print(f"💡 Ejecuta manualmente: {os.path.join(ffmpeg_dir, 'ffmpeg.exe')}")
                
        else:
            print("❌ No se encontró ffmpeg.exe")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Solución manual:")
        print("1. Descarga FFmpeg desde: https://ffmpeg.org/download.html")
        print("2. Extrae la carpeta 'bin' a tu proyecto")
        print("3. Renombra la carpeta a 'ffmpeg'")

if __name__ == "__main__":
    descargar_ffmpeg_portable()