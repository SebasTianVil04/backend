import os
import urllib.request
import zipfile
import tempfile
import shutil


def descargar_ffmpeg_portable():
    print("🚀 Descargando FFmpeg portable...")

    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "ffmpeg.zip")
    extract_dir = os.path.join(temp_dir, "ffmpeg_extract")

    try:
        print("📥 Descargando...")
        urllib.request.urlretrieve(url, zip_path)

        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1024:
            print("❌ La descarga falló o el archivo está incompleto")
            return

        print("📦 Extrayendo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        ffmpeg_exe = None
        for root, dirs, files in os.walk(extract_dir):
            if 'ffmpeg.exe' in files:
                ffmpeg_exe = os.path.join(root, 'ffmpeg.exe')
                break

        if ffmpeg_exe:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            ffmpeg_dir = os.path.join(project_dir, "ffmpeg")
            os.makedirs(ffmpeg_dir, exist_ok=True)

            shutil.copy2(ffmpeg_exe, os.path.join(ffmpeg_dir, "ffmpeg.exe"))

            print(f"✅ FFmpeg instalado en: {ffmpeg_dir}")
            print("🔧 Agregando al PATH de esta sesión...")

            os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ['PATH']

            result = os.system('ffmpeg -version')
            if result == 0:
                print("🎉 FFmpeg funciona correctamente!")
            else:
                print("⚠️ FFmpeg instalado pero no funciona automáticamente")
                print(f"💡 Ejecuta manualmente: {os.path.join(ffmpeg_dir, 'ffmpeg.exe')}")
            print("ℹ️ Este cambio de PATH solo dura mientras corra este proceso; agrega la carpeta al PATH del sistema para que persista.")

        else:
            print("❌ No se encontró ffmpeg.exe")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Solución manual:")
        print("1. Descarga FFmpeg desde: https://ffmpeg.org/download.html")
        print("2. Extrae la carpeta 'bin' a tu proyecto")
        print("3. Renombra la carpeta a 'ffmpeg'")

    finally:
        for ruta in (zip_path, extract_dir):
            try:
                if os.path.isfile(ruta):
                    os.remove(ruta)
                elif os.path.isdir(ruta):
                    shutil.rmtree(ruta, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    descargar_ffmpeg_portable()