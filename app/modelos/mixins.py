from sqlalchemy import Column, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, declared_attr


class AuditoriaMixin:

    @declared_attr
    def creado_por_id(cls):
        return Column(Integer, ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True, index=True)

    @declared_attr
    def actualizado_por_id(cls):
        return Column(Integer, ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True, index=True)

    @declared_attr
    def fecha_creacion(cls):
        return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    @declared_attr
    def fecha_actualizacion(cls):
        return Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    @declared_attr
    def creado_por(cls):
        kwargs = {
            "foreign_keys": [cls.creado_por_id],
            "lazy": "joined",
            "viewonly": True,
        }
        if cls.__name__ == "Usuario":
            kwargs["remote_side"] = [cls.id]
        return relationship("Usuario", **kwargs)

    @declared_attr
    def actualizado_por(cls):
        kwargs = {
            "foreign_keys": [cls.actualizado_por_id],
            "lazy": "joined",
            "viewonly": True,
        }
        if cls.__name__ == "Usuario":
            kwargs["remote_side"] = [cls.id]
        return relationship("Usuario", **kwargs)