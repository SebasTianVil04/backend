import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.utilidades.base_datos import obtener_bd
from app.dependencias.permisos import requiere_permiso
from app.modelos.usuario import Usuario
from app.modelos.ticket import TicketSoporte
from app.esquemas.ticket_schemas import (
    ComentarioIn,
    EstadoTicket,
    TicketCreadoOut,
    TicketUsuarioIn,
    TicketUsuarioOut,
)
from app.servicios import antispam_contacto as antispam
from app.servicios.servicio_tickets import (
    ESTADOS_OCULTOS_AL_SOLICITANTE,
    actualizar_ticket_admin,
    agregar_comentario,
    crear_ticket,
    obtener_ticket_propio_o_404,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tickets", tags=["Tickets"])


@router.post("", response_model=TicketCreadoOut, status_code=status.HTTP_201_CREATED)
def crear_ticket_usuario(
    datos: TicketUsuarioIn,
    request: Request,
    usuario_actual: Usuario = Depends(requiere_permiso("tickets.crear")),
    db: Session = Depends(obtener_bd),
):
    ip_h = antispam.hash_ip(antispam.obtener_ip_cliente(request))

    try:
        antispam.verificar_limites_usuario(db, usuario_actual.id, ip_h)
    except antispam.LimiteExcedido as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.detalle,
            headers={"Retry-After": str(e.reintentar_en)},
        )

    hash_c = antispam.hash_contenido(usuario_actual.email, datos.asunto, datos.descripcion)
    if antispam.buscar_duplicado(db, hash_c):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ya registraste un ticket idéntico recientemente")

    puntaje, motivos = antispam.calcular_puntaje_spam(datos.asunto, datos.descripcion, usuario_actual.email)
    en_cuarentena = puntaje >= antispam.UMBRAL_CUARENTENA
    if en_cuarentena:
        logger.warning("[TICKETS] Cuarentena usuario=%s puntaje=%d motivos=%s", usuario_actual.id, puntaje, motivos)

    ticket = crear_ticket(
        db,
        nombre=usuario_actual.nombre_usuario,
        email=usuario_actual.email,
        tipo=datos.tipo.value,
        servicio=datos.servicio.value,
        asunto=datos.asunto,
        descripcion=datos.descripcion,
        prioridad=datos.prioridad.value,
        origen="autenticado",
        usuario_id=usuario_actual.id,
        ip_hash=ip_h,
        hash_contenido=hash_c,
        puntaje_spam=puntaje,
        en_cuarentena=en_cuarentena,
    )

    return TicketCreadoOut(
        codigo=ticket.codigo,
        estado="nueva",
        mensaje="Ticket registrado correctamente",
    )


@router.get("/mis", response_model=List[TicketUsuarioOut])
def listar_mis_tickets(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    usuario_actual: Usuario = Depends(requiere_permiso("tickets.ver_propios")),
    db: Session = Depends(obtener_bd),
):
    tickets = (
        db.query(TicketSoporte)
        .filter(
            TicketSoporte.usuario_id == usuario_actual.id,
            TicketSoporte.estado.notin_(ESTADOS_OCULTOS_AL_SOLICITANTE),
        )
        .order_by(TicketSoporte.fecha_creacion.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [TicketUsuarioOut.model_validate(_visibles_copia(t)) for t in tickets]


def _visibles_copia(ticket: TicketSoporte) -> dict:
    return {
        "id": ticket.id,
        "codigo": ticket.codigo,
        "tipo": ticket.tipo,
        "servicio": ticket.servicio,
        "asunto": ticket.asunto,
        "descripcion": ticket.descripcion,
        "prioridad": ticket.prioridad,
        "estado": ticket.estado,
        "fecha_creacion": ticket.fecha_creacion,
        "fecha_actualizacion": ticket.fecha_actualizacion,
        "fecha_cierre": ticket.fecha_cierre,
        "comentarios": [
            {
                "autor_nombre": c.autor_nombre,
                "es_soporte": c.es_soporte,
                "mensaje": c.mensaje,
                "fecha_creacion": c.fecha_creacion,
            }
            for c in ticket.comentarios
            if not c.es_interno
        ],
    }


@router.get("/mis/{ticket_id}", response_model=TicketUsuarioOut)
def obtener_mi_ticket(
    ticket_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("tickets.ver_propios")),
    db: Session = Depends(obtener_bd),
):
    ticket = obtener_ticket_propio_o_404(db, ticket_id, usuario_actual)
    return TicketUsuarioOut.model_validate(_visibles_copia(ticket))


@router.post("/mis/{ticket_id}/comentarios", status_code=status.HTTP_201_CREATED)
def comentar_mi_ticket(
    ticket_id: int,
    datos: ComentarioIn,
    usuario_actual: Usuario = Depends(requiere_permiso("tickets.crear")),
    db: Session = Depends(obtener_bd),
):
    ticket = obtener_ticket_propio_o_404(db, ticket_id, usuario_actual)
    agregar_comentario(
        db, ticket, autor=usuario_actual, mensaje=datos.mensaje, es_soporte=False, es_interno=False
    )
    return {"exito": True, "mensaje": "Comentario agregado"}


@router.post("/mis/{ticket_id}/confirmar-cierre")
def confirmar_cierre_mi_ticket(
    ticket_id: int,
    usuario_actual: Usuario = Depends(requiere_permiso("tickets.crear")),
    db: Session = Depends(obtener_bd),
):
    ticket = obtener_ticket_propio_o_404(db, ticket_id, usuario_actual)
    if ticket.estado != "resuelta":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se puede confirmar el cierre de un ticket resuelto",
        )
    actualizar_ticket_admin(
        db, ticket, estado=EstadoTicket.cerrada, prioridad=None, asignado_a_id=None
    )
    return {"exito": True, "mensaje": "Ticket cerrado"}
