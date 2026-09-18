from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, ForeignKey, Table
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..utilidades.base_datos import Base

rol_permisos = Table(
    "rol_permisos",
    Base.metadata,
    Column("rol_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permiso_id", Integer, ForeignKey("permisos.id", ondelete="CASCADE"), primary_key=True),
)


class Rol(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    codigo = Column(String(50), unique=True, index=True, nullable=False)
    nombre = Column(String(100), nullable=False)
    descripcion = Column(Text, nullable=True)
    es_sistema = Column(Boolean, default=False, nullable=False)
    activo = Column(Boolean, default=True, nullable=False)
    creado_en = Column(DateTime(timezone=True), server_default=func.now())
    actualizado_en = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    permisos = relationship("Permiso", secondary=rol_permisos, back_populates="roles")
    usuarios = relationship("Usuario", back_populates="rol")

    def tiene_permiso(self, codigo: str) -> bool:
        return any(p.codigo == codigo for p in self.permisos)

    def __repr__(self):
        return f"<Rol(codigo={self.codigo})>"


class Permiso(Base):
    __tablename__ = "permisos"

    id = Column(Integer, primary_key=True, index=True)
    codigo = Column(String(100), unique=True, index=True, nullable=False)
    modulo = Column(String(50), nullable=False, index=True)
    descripcion = Column(String(255), nullable=False)
    creado_en = Column(DateTime(timezone=True), server_default=func.now())

    roles = relationship("Rol", secondary=rol_permisos, back_populates="permisos")

    def __repr__(self):
        return f"<Permiso(codigo={self.codigo})>"