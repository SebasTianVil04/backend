import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.utilidades.base_datos import obtener_bd
from app.modelos.ticket import TicketSoporte
from app.esquemas.ticket_schemas import (
    ContactoPublicoIn,
    SeguimientoOut,
    TicketCreadoOut,
    TokenFormularioOut,
)
from app.servicios import antispam_contacto as antispam
from app.servicios.servicio_tickets import ESTADOS_OCULTOS_AL_SOLICITANTE, crear_ticket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contacto", tags=["Contacto"])

_limitador_token = antispam.LimitadorMemoria(maximo=30, ventana_segundos=60)
_limitador_seguimiento = antispam.LimitadorMemoria(maximo=10, ventana_segundos=600)

MENSAJE_EXITO = "Recibimos tu solicitud. Guarda tu código de seguimiento."


def _respuesta_falsa() -> TicketCreadoOut:
    return TicketCreadoOut(
        codigo=f"TCK-{secrets.randbelow(9000) + 1000}-{secrets.randbelow(900000) + 100000}",
        estado="nueva",
        mensaje=MENSAJE_EXITO,
    )


@router.get("/token", response_model=TokenFormularioOut)
def obtener_token_formulario(request: Request):
    ip = antispam.obtener_ip_cliente(request)
    if not _limitador_token.permitir(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiadas solicitudes. Espera un momento.",
            headers={"Retry-After": "60"},
        )
    return TokenFormularioOut(
        token=antispam.emitir_token_formulario(),
        espera_minima_segundos=antispam.ESPERA_MINIMA_SEGUNDOS,
        turnstile_requerido=antispam.turnstile_requerido(),
    )


@router.post("", response_model=TicketCreadoOut, status_code=status.HTTP_201_CREATED)
def enviar_contacto(
    datos: ContactoPublicoIn,
    request: Request,
    db: Session = Depends(obtener_bd),
):
    ip = antispam.obtener_ip_cliente(request)
    ip_h = antispam.hash_ip(ip)

    if datos.sitio_web and datos.sitio_web.strip():
        logger.warning("[CONTACTO] Honeypot activado ip_hash=%s", ip_h[:12])
        return _respuesta_falsa()

    try:
        antispam.validar_token_formulario(datos.token_formulario)
    except antispam.TokenInvalido:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formulario inválido. Recarga la página.")
    except antispam.TokenExpirado:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El formulario expiró. Recarga la página e inténtalo de nuevo.")
    except antispam.FormularioMuyRapido:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Espera unos segundos antes de enviar el formulario.",
            headers={"Retry-After": str(antispam.ESPERA_MINIMA_SEGUNDOS)},
        )

    if not antispam.verificar_turnstile(datos.turnstile_token, ip):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se pudo verificar que eres una persona. Inténtalo de nuevo.")

    email_c = antispam.email_canonico(datos.email)

    try:
        antispam.verificar_limites_publicos(db, ip_h, datos.email)
        if email_c != datos.email:
            antispam.verificar_limites_publicos(db, ip_h, email_c)
    except antispam.LimiteExcedido as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.detalle,
            headers={"Retry-After": str(e.reintentar_en)},
        )

    hash_c = antispam.hash_contenido(datos.email, datos.asunto, datos.descripcion)
    if antispam.buscar_duplicado(db, hash_c):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ya recibimos una solicitud idéntica. Te responderemos pronto.")

    puntaje, motivos = antispam.calcular_puntaje_spam(datos.asunto, datos.descripcion, datos.email)
    en_cuarentena = puntaje >= antispam.UMBRAL_CUARENTENA
    if en_cuarentena:
        logger.warning("[CONTACTO] Cuarentena puntaje=%d motivos=%s ip_hash=%s", puntaje, motivos, ip_h[:12])

    ticket = crear_ticket(
        db,
        nombre=datos.nombre,
        email=datos.email,
        tipo=datos.tipo.value,
        servicio=datos.servicio.value,
        asunto=datos.asunto,
        descripcion=datos.descripcion,
        prioridad="media",
        origen="publico",
        usuario_id=None,
        ip_hash=ip_h,
        hash_contenido=hash_c,
        puntaje_spam=puntaje,
        en_cuarentena=en_cuarentena,
    )

    return TicketCreadoOut(codigo=ticket.codigo, estado="nueva", mensaje=MENSAJE_EXITO)


@router.get("/seguimiento", response_model=SeguimientoOut)
def consultar_seguimiento(
    request: Request,
    codigo: str = Query(..., min_length=8, max_length=20, pattern=r"^[A-Za-z]{3}-\d{4}-\d{6}$"),
    email: str = Query(..., min_length=6, max_length=255),
    db: Session = Depends(obtener_bd),
):
    ip = antispam.obtener_ip_cliente(request)
    if not _limitador_seguimiento.permitir(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiadas consultas. Inténtalo en unos minutos.",
            headers={"Retry-After": "600"},
        )

    ticket: Optional[TicketSoporte] = (
        db.query(TicketSoporte)
        .filter(
            TicketSoporte.codigo == codigo.upper(),
            TicketSoporte.email == email.strip().lower(),
            TicketSoporte.estado.notin_(ESTADOS_OCULTOS_AL_SOLICITANTE),
        )
        .first()
    )
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No se encontró un ticket con esos datos")

    return SeguimientoOut(
        codigo=ticket.codigo,
        tipo=ticket.tipo,
        servicio=ticket.servicio,
        asunto=ticket.asunto,
        estado=ticket.estado,
        fecha_creacion=ticket.fecha_creacion,
        fecha_actualizacion=ticket.fecha_actualizacion,
        comentarios=[c for c in ticket.comentarios if not c.es_interno],
    )
