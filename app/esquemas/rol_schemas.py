from typing import List, Optional

from pydantic import BaseModel, Field


class PermisoRespuesta(BaseModel):
    id: int
    codigo: str
    modulo: str
    descripcion: str

    class Config:
        from_attributes = True


class PermisoCrear(BaseModel):
    codigo: str = Field(..., min_length=3, max_length=80, pattern=r"^[a-z0-9_.]+$")
    modulo: str = Field(..., min_length=2, max_length=50, pattern=r"^[a-z0-9_]+$")
    descripcion: str = Field(..., min_length=3, max_length=255)


class PermisoActualizar(BaseModel):
    modulo: Optional[str] = Field(None, min_length=2, max_length=50, pattern=r"^[a-z0-9_]+$")
    descripcion: Optional[str] = Field(None, min_length=3, max_length=255)


class RolRespuesta(BaseModel):
    id: int
    codigo: str
    nombre: str
    descripcion: Optional[str] = None
    es_sistema: bool
    activo: bool
    permisos: List[PermisoRespuesta] = []

    class Config:
        from_attributes = True


class RolCrear(BaseModel):
    codigo: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-z0-9_]+$")
    nombre: str = Field(..., min_length=2, max_length=100)
    descripcion: Optional[str] = Field(None, max_length=500)
    permisos_ids: List[int] = []


class RolActualizar(BaseModel):
    nombre: Optional[str] = Field(None, min_length=2, max_length=100)
    descripcion: Optional[str] = Field(None, max_length=500)
    activo: Optional[bool] = None


class AsignarPermisosRequest(BaseModel):
    permisos_ids: List[int]