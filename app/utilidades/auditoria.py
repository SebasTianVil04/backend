from sqlalchemy import event
from sqlalchemy.orm import Session

from ..modelos.mixins import AuditoriaMixin

CLAVE_USUARIO_ACTUAL = "usuario_actual_id"


def marcar_usuario_actual(sesion: Session, usuario_id):
    sesion.info[CLAVE_USUARIO_ACTUAL] = usuario_id


def obtener_usuario_actual_de_sesion(sesion: Session):
    return sesion.info.get(CLAVE_USUARIO_ACTUAL)


@event.listens_for(Session, "before_flush")
def _asignar_campos_auditoria(session, flush_context, instances):
    usuario_id = session.info.get(CLAVE_USUARIO_ACTUAL)
    if usuario_id is None:
        return

    for obj in session.new:
        if isinstance(obj, AuditoriaMixin):
            if getattr(obj, "creado_por_id", None) is None:
                obj.creado_por_id = usuario_id
            obj.actualizado_por_id = usuario_id

    for obj in session.dirty:
        if isinstance(obj, AuditoriaMixin) and session.is_modified(obj, include_collections=False):
            obj.actualizado_por_id = usuario_id