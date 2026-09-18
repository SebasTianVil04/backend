from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship, backref
from ..utilidades.base_datos import Base


class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(100), nullable=False)
    icon = Column(String(20), nullable=True)
    ruta = Column(String(200), nullable=True)
    permiso_codigo = Column(String(100), nullable=True)
    orden = Column(Integer, default=0)
    activo = Column(Boolean, default=True)
    padre_id = Column(Integer, ForeignKey("menu_items.id"), nullable=True)

    hijos = relationship(
        "MenuItem",
        backref=backref("padre", remote_side=[id]),
        cascade="all, delete-orphan",
        order_by="MenuItem.orden",
    )