from typing import List, Optional
from pydantic import BaseModel


class PermisoRespuesta(BaseModel):
    id: int
    codigo: str
    modulo: str
    descripcion: str

    class Config:
        orm_mode = True


class RolRespuesta(BaseModel):
    id: int
    codigo: str
    nombre: str
    descripcion: Optional[str] = None
    es_sistema: bool
    activo: bool
    permisos: List[PermisoRespuesta] = []

    class Config:
        orm_mode = True


class RolCrear(BaseModel):
    codigo: str
    nombre: str
    descripcion: Optional[str] = None
    permisos_ids: List[int] = []


class RolActualizar(BaseModel):
    nombre: Optional[str] = None
    descripcion: Optional[str] = None
    activo: Optional[bool] = None


class AsignarPermisosRequest(BaseModel):
    permisos_ids: List[int]