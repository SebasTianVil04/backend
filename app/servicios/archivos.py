import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles
from fastapi import HTTPException, UploadFile, status
from PIL import Image

from ..utilidades.configuracion import configuracion

PATRON_CATEGORIA = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TAMANO_BLOQUE = 1024 * 1024


class ArchivoService:
    def __init__(self):
        self.upload_dir = Path(configuracion.upload_dir).resolve()
        self.temp_dir = Path(configuracion.temp_dir).resolve()
        self.max_file_size = configuracion.max_file_size

        self.extensiones_imagen = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        self.extensiones_video = {".mp4", ".avi", ".mov", ".wmv", ".flv"}
        self.extensiones_permitidas = self.extensiones_imagen | self.extensiones_video

        self._crear_directorios()

    def _crear_directorios(self):
        directorios = [
            self.upload_dir,
            self.temp_dir,
            self.upload_dir / "imagenes_entrenamiento",
            self.upload_dir / "videos_lecciones",
            self.upload_dir / "avatares_usuarios",
        ]
        for directorio in directorios:
            directorio.mkdir(parents=True, exist_ok=True)

    def _validar_archivo(self, archivo: UploadFile) -> None:
        if not archivo.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo debe tener un nombre válido",
            )

        extension = Path(archivo.filename).suffix.lower()
        if extension not in self.extensiones_permitidas:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Extensión no permitida. Usa: {', '.join(sorted(self.extensiones_permitidas))}",
            )

        if getattr(archivo, "size", None) and archivo.size > self.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"El archivo es muy grande. Máximo {self.max_file_size / (1024 * 1024):.0f}MB permitidos",
            )

    def _generar_nombre_unico(self, nombre_original: str) -> str:
        extension = Path(nombre_original).suffix.lower()
        return f"{uuid.uuid4().hex}{extension}"

    def _validar_categoria(self, categoria: str) -> str:
        if not PATRON_CATEGORIA.fullmatch(categoria or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Categoría inválida",
            )
        return categoria

    def _ruta_segura(self, base: Path, *partes: str) -> Path:
        ruta = base.joinpath(*partes).resolve()
        if not ruta.is_relative_to(base.resolve()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ruta inválida",
            )
        return ruta

    async def _guardar_limitado(self, archivo: UploadFile, destino: Path) -> None:
        total = 0
        try:
            async with aiofiles.open(destino, "wb") as f:
                while True:
                    bloque = await archivo.read(TAMANO_BLOQUE)
                    if not bloque:
                        break
                    total += len(bloque)
                    if total > self.max_file_size:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"El archivo es muy grande. Máximo {self.max_file_size / (1024 * 1024):.0f}MB permitidos",
                        )
                    await f.write(bloque)
        except BaseException:
            if destino.exists():
                os.remove(destino)
            raise

    def _firma_video_valida(self, ruta: Path, extension: str) -> bool:
        with open(ruta, "rb") as f:
            cabecera = f.read(16)
        if extension in {".mp4", ".mov"}:
            return cabecera[4:8] == b"ftyp" or cabecera[4:8] in {b"moov", b"mdat", b"free", b"wide"}
        if extension in {".avi"}:
            return cabecera[:4] == b"RIFF" and cabecera[8:12] == b"AVI "
        if extension == ".wmv":
            return cabecera[:4] == b"\x30\x26\xb2\x75"
        if extension == ".flv":
            return cabecera[:3] == b"FLV"
        return False

    async def subir_imagen_entrenamiento(
        self,
        archivo: UploadFile,
        categoria: str = "general",
    ) -> Tuple[str, str]:
        self._validar_archivo(archivo)
        categoria = self._validar_categoria(categoria)

        extension = Path(archivo.filename).suffix.lower()
        if extension not in self.extensiones_imagen:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se permiten archivos de imagen para entrenamiento",
            )

        nombre_archivo = self._generar_nombre_unico(archivo.filename)
        directorio_categoria = self._ruta_segura(self.upload_dir, "imagenes_entrenamiento", categoria)
        directorio_categoria.mkdir(parents=True, exist_ok=True)

        ruta_completa = directorio_categoria / nombre_archivo
        ruta_relativa = f"imagenes_entrenamiento/{categoria}/{nombre_archivo}"

        try:
            await self._guardar_limitado(archivo, ruta_completa)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al guardar archivo",
            )

        try:
            with Image.open(ruta_completa) as img:
                img.verify()
        except Exception:
            if ruta_completa.exists():
                os.remove(ruta_completa)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo no es una imagen válida",
            )

        return ruta_relativa, str(ruta_completa)

    async def subir_video_leccion(self, archivo: UploadFile) -> Tuple[str, str]:
        self._validar_archivo(archivo)

        extension = Path(archivo.filename).suffix.lower()
        if extension not in self.extensiones_video:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se permiten archivos de video",
            )

        nombre_archivo = self._generar_nombre_unico(archivo.filename)
        ruta_completa = self._ruta_segura(self.upload_dir, "videos_lecciones", nombre_archivo)
        ruta_relativa = f"videos_lecciones/{nombre_archivo}"

        try:
            await self._guardar_limitado(archivo, ruta_completa)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al guardar archivo",
            )

        if not self._firma_video_valida(ruta_completa, extension):
            if ruta_completa.exists():
                os.remove(ruta_completa)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo no es un video válido",
            )

        return ruta_relativa, str(ruta_completa)

    def eliminar_archivo(self, ruta_relativa: str) -> bool:
        try:
            ruta_completa = self._ruta_segura(self.upload_dir, ruta_relativa)
            if ruta_completa.is_file():
                os.remove(ruta_completa)
                return True
            return False
        except Exception:
            return False

    def obtener_ruta_completa(self, ruta_relativa: str) -> str:
        return str(self._ruta_segura(self.upload_dir, ruta_relativa))

    def archivo_existe(self, ruta_relativa: str) -> bool:
        try:
            return self._ruta_segura(self.upload_dir, ruta_relativa).exists()
        except HTTPException:
            return False

    def obtener_info_archivo(self, ruta_relativa: str) -> Optional[dict]:
        try:
            ruta_completa = self._ruta_segura(self.upload_dir, ruta_relativa)
            if not ruta_completa.exists():
                return None

            stat = ruta_completa.stat()
            return {
                "nombre": ruta_completa.name,
                "tamaño": stat.st_size,
                "fecha_modificacion": stat.st_mtime,
                "es_imagen": ruta_completa.suffix.lower() in self.extensiones_imagen,
                "es_video": ruta_completa.suffix.lower() in self.extensiones_video,
            }
        except Exception:
            return None

    def listar_archivos_entrenamiento(self, categoria: Optional[str] = None) -> List[dict]:
        try:
            directorio = self.upload_dir / "imagenes_entrenamiento"
            if categoria:
                directorio = self._ruta_segura(directorio, self._validar_categoria(categoria))

            archivos = []
            if directorio.exists():
                for archivo in directorio.rglob("*"):
                    if archivo.is_file():
                        info = self.obtener_info_archivo(str(archivo.relative_to(self.upload_dir)))
                        if info:
                            archivos.append(info)
            return archivos
        except Exception:
            return []


archivo_service = ArchivoService()