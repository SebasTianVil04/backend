import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException, status
from PIL import Image
import aiofiles

from ..utilidades.configuracion import configuracion

class ArchivoService:
    """Servicio para manejo de archivos (imágenes, videos, etc.)"""
    
    def __init__(self):
        self.upload_dir = Path(configuracion.upload_dir)
        self.temp_dir = Path(configuracion.temp_dir)
        self.max_file_size = configuracion.max_file_size
        
        # Extensiones permitidas
        self.extensiones_imagen = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        self.extensiones_video = {".mp4", ".avi", ".mov", ".wmv", ".flv"}
        self.extensiones_permitidas = self.extensiones_imagen | self.extensiones_video
        
        # Crear directorios si no existen
        self._crear_directorios()
    
    def _crear_directorios(self):
        """Crear directorios necesarios para almacenar archivos"""
        directorios = [
            self.upload_dir,
            self.temp_dir,
            self.upload_dir / "imagenes_entrenamiento",
            self.upload_dir / "videos_lecciones",
            self.upload_dir / "avatares_usuarios"
        ]
        
        for directorio in directorios:
            directorio.mkdir(parents=True, exist_ok=True)
    
    def _validar_archivo(self, archivo: UploadFile) -> None:
        """Validar que el archivo cumple con los requisitos"""
        if not archivo.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo debe tener un nombre válido"
            )
        
        # Validar extensión
        extension = Path(archivo.filename).suffix.lower()
        if extension not in self.extensiones_permitidas:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Extensión no permitida. Usa: {', '.join(self.extensiones_permitidas)}"
            )
        
        # Validar tamaño (si está disponible)
        if hasattr(archivo, 'size') and archivo.size and archivo.size > self.max_file_size:
            tamaño_mb = self.max_file_size / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"El archivo es muy grande. Máximo {tamaño_mb}MB permitidos"
            )
    
    def _generar_nombre_unico(self, nombre_original: str) -> str:
        """Generar un nombre único para el archivo"""
        extension = Path(nombre_original).suffix.lower()
        nombre_unico = f"{uuid.uuid4()}{extension}"
        return nombre_unico
    
    async def subir_imagen_entrenamiento(
        self, 
        archivo: UploadFile,
        categoria: str = "general"
    ) -> Tuple[str, str]:

        self._validar_archivo(archivo)
        
        # Verificar que sea imagen
        extension = Path(archivo.filename).suffix.lower()
        if extension not in self.extensiones_imagen:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se permiten archivos de imagen para entrenamiento"
            )
        
        # Generar nombre único
        nombre_archivo = self._generar_nombre_unico(archivo.filename)
        
        # Crear subdirectorio por categoría
        directorio_categoria = self.upload_dir / "imagenes_entrenamiento" / categoria
        directorio_categoria.mkdir(exist_ok=True)
        
        # Ruta completa del archivo
        ruta_completa = directorio_categoria / nombre_archivo
        ruta_relativa = f"imagenes_entrenamiento/{categoria}/{nombre_archivo}"
        
        # Guardar archivo
        try:
            async with aiofiles.open(ruta_completa, 'wb') as f:
                contenido = await archivo.read()
                await f.write(contenido)
            
            # Validar que la imagen se puede abrir
            try:
                with Image.open(ruta_completa) as img:
                    img.verify()
            except Exception:
                # Eliminar archivo si no es una imagen válida
                os.remove(ruta_completa)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El archivo no es una imagen válida"
                )
            
            return ruta_relativa, str(ruta_completa)
            
        except Exception as e:
            if os.path.exists(ruta_completa):
                os.remove(ruta_completa)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al guardar archivo: {str(e)}"
            )
    
    async def subir_video_leccion(self, archivo: UploadFile) -> Tuple[str, str]:
        """
        Subir un video para una lección
        
        Args:
            archivo: Archivo de video
            
        Returns:
            Tuple con (ruta_relativa, ruta_completa)
        """
        self._validar_archivo(archivo)
        
        # Verificar que sea video
        extension = Path(archivo.filename).suffix.lower()
        if extension not in self.extensiones_video:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Solo se permiten archivos de video"
            )
        
        # Generar nombre único
        nombre_archivo = self._generar_nombre_unico(archivo.filename)
        
        # Ruta completa del archivo
        ruta_completa = self.upload_dir / "videos_lecciones" / nombre_archivo
        ruta_relativa = f"videos_lecciones/{nombre_archivo}"
        
        # Guardar archivo
        try:
            async with aiofiles.open(ruta_completa, 'wb') as f:
                contenido = await archivo.read()
                await f.write(contenido)
            
            return ruta_relativa, str(ruta_completa)
            
        except Exception as e:
            if os.path.exists(ruta_completa):
                os.remove(ruta_completa)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al guardar archivo: {str(e)}"
            )
    
    def eliminar_archivo(self, ruta_relativa: str) -> bool:
        try:
            ruta_completa = self.upload_dir / ruta_relativa
            if ruta_completa.exists():
                os.remove(ruta_completa)
                return True
            return False
        except Exception:
            return False
    
    def obtener_ruta_completa(self, ruta_relativa: str) -> str:
        """Obtener la ruta completa de un archivo"""
        return str(self.upload_dir / ruta_relativa)
    
    def archivo_existe(self, ruta_relativa: str) -> bool:
        """Verificar si un archivo existe"""
        ruta_completa = self.upload_dir / ruta_relativa
        return ruta_completa.exists()
    
    def obtener_info_archivo(self, ruta_relativa: str) -> Optional[dict]:
        """Obtener información de un archivo"""
        try:
            ruta_completa = self.upload_dir / ruta_relativa
            if not ruta_completa.exists():
                return None
            
            stat = ruta_completa.stat()
            return {
                "nombre": ruta_completa.name,
                "tamaño": stat.st_size,
                "fecha_modificacion": stat.st_mtime,
                "es_imagen": ruta_completa.suffix.lower() in self.extensiones_imagen,
                "es_video": ruta_completa.suffix.lower() in self.extensiones_video
            }
        except Exception:
            return None
    
    def listar_archivos_entrenamiento(self, categoria: Optional[str] = None) -> List[dict]:
        """Listar archivos de entrenamiento"""
        try:
            directorio = self.upload_dir / "imagenes_entrenamiento"
            if categoria:
                directorio = directorio / categoria
            
            archivos = []
            if directorio.exists():
                for archivo in directorio.rglob("*"):
                    if archivo.is_file():
                        info = self.obtener_info_archivo(
                            str(archivo.relative_to(self.upload_dir))
                        )
                        if info:
                            archivos.append(info)
            
            return archivos
        except Exception:
            return []

# Instancia global del servicio
archivo_service = ArchivoService() 
