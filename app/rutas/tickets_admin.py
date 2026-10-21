from http.client import HTTPException
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.utilidades.base_datos import obtener_bd
from app.dependencias.permisos import requiere_permiso
from app.modelos.usuario import Usuario
from app.modelos.ticket import TicketSoporte
from app.esquemas.ticket_schemas import (
    ComentarioIn,
    ComentarioOut,
    EstadoTicket,
    ServicioTicket,
    TicketAdminUpdate,
    TicketDetalleOut,
    TicketResumenOut,
    TipoTicket,
)
from app.servicios.servicio_tickets import (
    ESTADOS_OCULTOS_AL_SOLICITANTE,
    actualizar_ticket_admin,
    agregar_comentario,
    eliminar_ticket,
    obtener_ticket_o_404,
)

router = APIRouter(prefix="/admin/tickets", tags=["Tickets Admin"])


@router.get("", response_model=List[TicketResumenOut])
def listar_tickets_admin(
    estado: Optional[EstadoTicket] = None,
    servicio: Optional[ServicioTicket] = None,
    tipo: Optional[TipoTicket] = None,
    asignado_a_id: Optional[int] = Query(None, ge=1),
    q: Optional[str] = Query(None, max_length=100),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.tickets.gestionar")),
    db: Session = Depends(obtener_bd),
):
    consulta = db.query(TicketSoporte)

    if estado:
        consulta = consulta.filter(TicketSoporte.estado == estado.value)
    else:
        consulta = consulta.filter(TicketSoporte.estado.notin_(ESTADOS_OCULTOS_AL_SOLICITANTE))

    if servicio:
        consulta = consulta.filter(TicketSoporte.servicio == servicio.value)
    if tipo:
        consulta = consulta.filter(TicketSoporte.tipo == tipo.value)
    if asignado_a_id:
        consulta = consulta.filter(TicketSoporte.asignado_a_id == asignado_a_id)
    if q and q.strip():
        patron = f"%{q.strip().replace('%', '').replace('_', '')}%"
        consulta = consulta.filter(
            or_(
                TicketSoporte.codigo.ilike(patron),
                TicketSoporte.email.ilike(patron),
                TicketSoporte.asunto.ilike(patron),
            )
        )

    return consulta.order_by(TicketSoporte.fecha_creacion.desc()).offset(skip).limit(limit).all()


@router.get("/{ticket_id}", response_model=TicketDetalleOut)
def obtener_ticket_admin(
    ticket_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.tickets.gestionar")),
    db: Session = Depends(obtener_bd),
):
    return obtener_ticket_o_404(db, ticket_id)


@router.patch("/{ticket_id}", response_model=TicketDetalleOut)
def actualizar_ticket(
    ticket_id: int,
    datos: TicketAdminUpdate,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.tickets.gestionar")),
    db: Session = Depends(obtener_bd),
):
    ticket = obtener_ticket_o_404(db, ticket_id)
    return actualizar_ticket_admin(
        db,
        ticket,
        estado=datos.estado,
        prioridad=datos.prioridad.value if datos.prioridad else None,
        asignado_a_id=datos.asignado_a_id,
    )


@router.post("/{ticket_id}/comentarios", response_model=ComentarioOut, status_code=201)
def comentar_ticket_admin(
    ticket_id: int,
    datos: ComentarioIn,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.tickets.gestionar")),
    db: Session = Depends(obtener_bd),
):
    ticket = obtener_ticket_o_404(db, ticket_id)
    return agregar_comentario(
        db,
        ticket,
        autor=usuario_actual,
        mensaje=datos.mensaje,
        es_soporte=True,
        es_interno=datos.es_interno,
    )

@router.delete("/{ticket_id}", status_code=204)
def eliminar_ticket_admin(
    ticket_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("admin.tickets.gestionar")),
    db: Session = Depends(obtener_bd),
):
    if not usuario_actual.rol or usuario_actual.rol.codigo != "admin":
        raise HTTPException(status_code=403, detail="Solo el administrador puede eliminar tickets")
    ticket = obtener_ticket_o_404(db, ticket_id)
    eliminar_ticket(db, ticket)