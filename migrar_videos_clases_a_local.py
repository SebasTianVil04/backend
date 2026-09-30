import sys
import uuid

from googleapiclient.http import MediaIoBaseDownload
from sqlalchemy import select, update

from app.modelos.clase import Clase, TipoVideo
from app.servicios.drive_service import get_drive_service
from app.utilidades import archivos_firmados as firmas
from app.utilidades.base_datos import SessionLocal

TABLA = Clase.__table__
MIME_A_EXTENSION = {"video/mp4": "mp4", "video/webm": "webm"}
TAMANO_BLOQUE = 8 * 1024 * 1024


def metadatos(video_id: str) -> dict:
    return (
        get_drive_service()
        .files()
        .get(fileId=video_id, fields="name,mimeType,size", supportsAllDrives=True)
        .execute()
    )


def descargar(video_id: str, destino) -> None:
    solicitud = get_drive_service().files().get_media(fileId=video_id, supportsAllDrives=True)
    with open(destino, "wb") as salida:
        descargador = MediaIoBaseDownload(salida, solicitud, chunksize=TAMANO_BLOQUE)
        terminado = False
        while not terminado:
            _, terminado = descargador.next_chunk()


def main() -> None:
    ejecutar = "--ejecutar" in sys.argv
    directorio = firmas.RAICES["videos_lecciones"]["ruta"]
    directorio.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    migradas = 0
    incompatibles = []
    inaccesibles = []
    fallidas = []

    try:
        filas = db.execute(
            select(TABLA.c.id, TABLA.c.titulo, TABLA.c.video_id)
            .where(
                TABLA.c.tipo_video == TipoVideo.GOOGLE_DRIVE,
                TABLA.c.video_id.isnot(None),
                TABLA.c.video_archivo.is_(None),
            )
            .order_by(TABLA.c.id)
        ).all()

        print(f"Clases con video de Drive: {len(filas)}")
        print("Modo: EJECUCIÓN" if ejecutar else "Modo: SIMULACIÓN (usa --ejecutar para aplicar)")

        for fila in filas:
            try:
                meta = metadatos(fila.video_id)
            except Exception as e:
                inaccesibles.append(fila.id)
                print(f"[{fila.id}] {fila.titulo}: no se puede acceder al archivo de Drive ({e})")
                continue

            extension = MIME_A_EXTENSION.get(meta.get("mimeType", ""))
            tamano = int(meta.get("size", 0))

            if extension is None:
                incompatibles.append(fila.id)
                print(f"[{fila.id}] {fila.titulo}: formato no compatible ({meta.get('mimeType')})")
                continue

            if not ejecutar:
                print(f"[{fila.id}] {fila.titulo}: se descargaría {meta.get('name')} ({tamano:,} bytes, {extension})")
                continue

            nombre = f"{uuid.uuid4().hex}.{extension}"
            destino = directorio / nombre
            parcial = directorio / f"{nombre}.part"

            try:
                descargar(fila.video_id, parcial)

                if tamano and parcial.stat().st_size != tamano:
                    raise RuntimeError("El tamaño descargado no coincide con el de Drive")

                parcial.replace(destino)

                db.execute(update(TABLA).where(TABLA.c.id == fila.id).values(video_archivo=nombre))
                db.commit()
            except Exception as e:
                db.rollback()
                for ruta in (parcial, destino):
                    if ruta.exists():
                        ruta.unlink()
                fallidas.append({"id": fila.id, "motivo": str(e)})
                print(f"[{fila.id}] {fila.titulo}: FALLÓ ({e})")
                continue

            migradas += 1
            print(f"[{fila.id}] {fila.titulo}: migrada")

        print()
        print(f"Migradas: {migradas}")
        print(f"Formato no compatible: {incompatibles}")
        print(f"Sin acceso en Drive: {inaccesibles}")
        print(f"Fallidas: {fallidas}")
    finally:
        db.close()


if __name__ == "__main__":
    main()