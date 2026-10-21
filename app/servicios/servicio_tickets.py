from datetime import datetime
from typing import Optional, Set

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..modelos.ticket import TicketSoporte, TicketComentario
from ..modelos.usuario import Usuario
from ..esquemas.ticket_schemas import EstadoTicket

ESTADOS_OCULTOS_AL_SOLICITANTE: Set[str] = {"cuarentena", "descartada"}
ESTADOS_TERMINALES: Set[str] = {"cerrada", "descartada"}

TRANSICIONES = {
    "cuarentena": {"nueva", "descartada"},
    "nueva": {"en_revision", "asignada", "rechazada", "descartada"},
    "en_revision": {"aprobada", "asignada", "rechazada", "esperando_usuario"},
    "aprobada": {"asignada"},
    "asignada": {"en_atencion", "esperando_usuario"},
    "en_atencion": {"resuelta", "esperando_usuario"},
    "esperando_usuario": {"en_atencion", "cerrada"},
    "resuelta": {"cerrada", "en_atencion"},
    "rechazada": {"cerrada"},
    "cerrada": set(),
    "descartada": set(),
}

MAX_COMENTARIOS_SOLICITANTE = 30


def crear_ticket(
    db: Session,
    *,
    nombre: str,
    email: str,
    tipo: str,
    servicio: str,
    asunto: str,
    descripcion: str,
    prioridad: str,
    origen: str,
    usuario_id: Optional[int],
    ip_hash: Optional[str],
    hash_contenido: Optional[str],
    puntaje_spam: int,
    en_cuarentena: bool,
) -> TicketSoporte:
    ticket = TicketSoporte(
        usuario_id=usuario_id,
        nombre=nombre,
        email=email,
        tipo=tipo,
        servicio=servicio,
        asunto=asunto,
        descripcion=descripcion,
        prioridad=prioridad,
        estado="cuarentena" if en_cuarentena else "nueva",
        origen=origen,
        puntaje_spam=puntaje_spam,
        ip_hash=ip_hash,
        hash_contenido=hash_contenido,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


def obtener_ticket_o_404(db: Session, ticket_id: int) -> TicketSoporte:
    ticket = db.query(TicketSoporte).filter(TicketSoporte.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket no encontrado")
    return ticket


def obtener_ticket_propio_o_404(db: Session, ticket_id: int, usuario: Usuario) -> TicketSoporte:
    ticket = (
        db.query(TicketSoporte)
        .filter(
            TicketSoporte.id == ticket_id,
            TicketSoporte.usuario_id == usuario.id,
            TicketSoporte.estado.notin_(ESTADOS_OCULTOS_AL_SOLICITANTE),
        )
        .first()
    )
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket no encontrado")
    return ticket


def agregar_comentario(
    db: Session,
    ticket: TicketSoporte,
    *,
    autor: Usuario,
    mensaje: str,
    es_soporte: bool,
    es_interno: bool,
) -> TicketComentario:
    if ticket.estado in ESTADOS_TERMINALES:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="El ticket ya está cerrado")

    if not es_soporte:
        total = (
            db.query(TicketComentario)
            .filter(TicketComentario.ticket_id == ticket.id, TicketComentario.es_soporte == False)
            .count()
        )
        if total >= MAX_COMENTARIOS_SOLICITANTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Se alcanzó el máximo de comentarios para este ticket",
            )

    comentario = TicketComentario(
        ticket_id=ticket.id,
        autor_id=autor.id,
        autor_nombre=autor.nombre_usuario,
        es_soporte=es_soporte,
        es_interno=es_interno and es_soporte,
        mensaje=mensaje,
    )
    db.add(comentario)

    if not es_soporte and ticket.estado == "esperando_usuario":
        ticket.estado = "en_atencion"

    ticket.fecha_actualizacion = datetime.utcnow()
    db.commit()
    db.refresh(comentario)
    return comentario


def actualizar_ticket_admin(
    db: Session,
    ticket: TicketSoporte,
    *,
    estado: Optional[EstadoTicket],
    prioridad: Optional[str],
    asignado_a_id: Optional[int],
) -> TicketSoporte:
    if asignado_a_id is not None:
        asignado = db.query(Usuario).filter(Usuario.id == asignado_a_id, Usuario.activo == True).first()
        if not asignado:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El responsable no existe o está inactivo")
        ticket.asignado_a_id = asignado_a_id

    if prioridad is not None:
        ticket.prioridad = prioridad

    if estado is not None:
        nuevo = estado.value
        if nuevo != ticket.estado:
            if nuevo not in TRANSICIONES.get(ticket.estado, set()):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No se puede pasar de '{ticket.estado}' a '{nuevo}'",
                )
            if nuevo == "asignada" and not ticket.asignado_a_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Debes indicar un responsable para asignar el ticket",
                )
            ticket.estado = nuevo
            if nuevo in ESTADOS_TERMINALES:
                ticket.fecha_cierre = datetime.utcnow()

    ticket.fecha_actualizacion = datetime.utcnow()
    db.commit()
    db.refresh(ticket)
    return ticket

def eliminar_ticket(db: Session, ticket: TicketSoporte) -> None:
    if ticket.estado not in ESTADOS_TERMINALES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Solo se pueden eliminar tickets cerrados o descartados",
        )
    db.query(TicketComentario).filter(TicketComentario.ticket_id == ticket.id).delete(synchronize_session=False)
    db.delete(ticket)
    db.commit()