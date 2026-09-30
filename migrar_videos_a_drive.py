import sys
from pathlib import Path

from sqlalchemy import select, update

from app.modelos.dataset import VideoDataset
from app.servicios.drive_service import (
    eliminar_archivo_de_drive,
    get_drive_service,
    subir_archivo_a_drive,
)
from app.utilidades.base_datos import SessionLocal

TABLA = VideoDataset.__table__


def tamano_en_drive(file_id: str) -> int:
    meta = (
        get_drive_service()
        .files()
        .get(fileId=file_id, fields="size", supportsAllDrives=True)
        .execute()
    )
    return int(meta.get("size", 0))


def main() -> None:
    ejecutar = "--ejecutar" in sys.argv
    db = SessionLocal()
    migrados = 0
    sin_archivo = []
    fallidos = []

    try:
        filas = db.execute(
            select(TABLA.c.id, TABLA.c.sena, TABLA.c.ruta_video)
            .where(TABLA.c.ruta_video.isnot(None), TABLA.c.drive_file_id.is_(None))
            .order_by(TABLA.c.id)
        ).all()

        print(f"Videos solo en local: {len(filas)}")
        print("Modo: EJECUCIÓN" if ejecutar else "Modo: SIMULACIÓN (usa --ejecutar para aplicar)")

        for fila in filas:
            ruta = Path(fila.ruta_video)

            if not ruta.is_file():
                sin_archivo.append(fila.id)
                print(f"[{fila.id}] {fila.sena}: el archivo local no existe ({ruta})")
                continue

            if not ejecutar:
                print(f"[{fila.id}] {fila.sena}: se subiría {ruta.name} ({ruta.stat().st_size:,} bytes)")
                continue

            file_id = None
            try:
                contenido = ruta.read_bytes()
                mime = "video/mp4" if ruta.suffix.lower() == ".mp4" else "video/webm"

                subida = subir_archivo_a_drive(contenido, ruta.name, mime, subcarpeta="videos_dataset")
                file_id = subida["id"]

                if tamano_en_drive(file_id) != len(contenido):
                    raise RuntimeError("El tamaño en Drive no coincide con el archivo local")

                db.execute(
                    update(TABLA)
                    .where(TABLA.c.id == fila.id)
                    .values(drive_file_id=file_id, drive_url=subida["url_ver"], ruta_video=None)
                )
                db.commit()
            except Exception as e:
                db.rollback()
                if file_id:
                    eliminar_archivo_de_drive(file_id)
                fallidos.append({"id": fila.id, "motivo": str(e)})
                print(f"[{fila.id}] {fila.sena}: FALLÓ ({e})")
                continue

            try:
                ruta.unlink()
            except OSError as e:
                print(f"[{fila.id}] subido, pero no se pudo borrar el local: {e}")

            migrados += 1
            print(f"[{fila.id}] {fila.sena}: migrado")

        print()
        print(f"Migrados: {migrados}")
        print(f"Sin archivo local: {sin_archivo}")
        print(f"Fallidos: {fallidos}")
    finally:
        db.close()


if __name__ == "__main__":
    main()