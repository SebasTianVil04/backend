import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
import unicodedata
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..modelos.ticket import TicketSoporte

logger = logging.getLogger(__name__)


def _entero_env(nombre: str, defecto: int) -> int:
    try:
        return int(os.getenv(nombre, defecto))
    except ValueError:
        return defecto


ESPERA_MINIMA_SEGUNDOS = _entero_env("CONTACTO_ESPERA_MIN", 3)
TOKEN_VIGENCIA_SEGUNDOS = _entero_env("CONTACTO_TOKEN_TTL", 7200)
LIMITE_IP_HORA = _entero_env("CONTACTO_LIMITE_IP_HORA", 5)
LIMITE_IP_DIA = _entero_env("CONTACTO_LIMITE_IP_DIA", 15)
LIMITE_EMAIL_HORA = _entero_env("CONTACTO_LIMITE_EMAIL_HORA", 3)
LIMITE_EMAIL_DIA = _entero_env("CONTACTO_LIMITE_EMAIL_DIA", 6)
LIMITE_USUARIO_DIA = _entero_env("CONTACTO_LIMITE_USUARIO_DIA", 10)
UMBRAL_CUARENTENA = _entero_env("CONTACTO_UMBRAL_CUARENTENA", 3)
VENTANA_DUPLICADOS_HORAS = _entero_env("CONTACTO_VENTANA_DUPLICADOS_H", 24)

TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

DOMINIOS_DESECHABLES = {
    "mailinator.com", "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "10minutemail.com", "10minutemail.net", "tempmail.com", "temp-mail.org",
    "temp-mail.io", "yopmail.com", "yopmail.net", "trashmail.com", "throwawaymail.com",
    "getnada.com", "nada.email", "sharklasers.com", "grr.la", "dispostable.com",
    "maildrop.cc", "fakeinbox.com", "mailnesia.com", "mohmal.com", "emailondeck.com",
    "tempail.com", "burnermail.io", "mintemail.com", "spamgourmet.com", "moakt.com",
    "tmpmail.org", "tmpmail.net", "discard.email", "mytemp.email", "inboxkitten.com",
}

PALABRAS_SPAM = (
    "bitcoin", "crypto", "criptomoneda", "casino", "apuestas", "viagra", "cialis",
    "forex", "ganar dinero", "dinero rapido", "dinero rápido", "inversion garantizada",
    "inversión garantizada", "prestamo", "préstamo", "seo", "backlinks", "posicionamiento web",
    "click aqui", "click aquí", "haz clic", "oferta limitada", "gratis ahora", "xxx",
    "porn", "telegram", "whatsapp", "work from home", "make money", "free money",
    "buy now", "limited offer", "congratulations", "you have won", "has ganado",
)

RE_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9\-]+\.(?:com|net|org|xyz|top|ru|cn|info|biz|shop|click|link|site|online)\b/?\S*")
RE_HTML = re.compile(r"(?i)<\s*/?\s*[a-z!][^>]*>|javascript:|data:text/html|\bon\w+\s*=")
RE_NOMBRE = re.compile(r"^[^\W\d_]+(?:[ '\-\.][^\W\d_]+)*$", re.UNICODE)
RE_EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}$")
RE_REPETIDO = re.compile(r"(.)\1{9,}")


def _dominios_bloqueados() -> set:
    extra = os.getenv("CONTACTO_DOMINIOS_BLOQUEADOS", "")
    return DOMINIOS_DESECHABLES | {d.strip().lower() for d in extra.split(",") if d.strip()}


def normalizar_texto(texto: str, multilinea: bool = False) -> str:
    texto = unicodedata.normalize("NFKC", texto or "")
    texto = "".join(
        c for c in texto
        if c in ("\n", "\t") or (unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    )
    if multilinea:
        lineas = [re.sub(r"[ \t]+", " ", l).strip() for l in texto.replace("\r", "\n").split("\n")]
        texto = re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()
    else:
        texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def validar_nombre(valor: str) -> str:
    valor = normalizar_texto(valor)
    if not 2 <= len(valor) <= 80:
        raise ValueError("El nombre debe tener entre 2 y 80 caracteres")
    if not RE_NOMBRE.match(valor):
        raise ValueError("El nombre solo puede contener letras, espacios, guion, apóstrofe y punto")
    if RE_URL.search(valor) or RE_REPETIDO.search(valor):
        raise ValueError("El nombre no es válido")
    return valor


def validar_email(valor: str) -> str:
    valor = normalizar_texto(valor).lower()
    if len(valor) > 255 or not RE_EMAIL.match(valor):
        raise ValueError("El correo electrónico no es válido")
    local, dominio = valor.rsplit("@", 1)
    if len(local) > 64 or ".." in valor or local.startswith(".") or local.endswith("."):
        raise ValueError("El correo electrónico no es válido")
    if dominio in _dominios_bloqueados():
        raise ValueError("Usa un correo electrónico permanente, no uno temporal")
    return valor


def email_canonico(email: str) -> str:
    local, dominio = email.lower().rsplit("@", 1)
    if dominio in ("gmail.com", "googlemail.com"):
        local = local.split("+", 1)[0].replace(".", "")
        dominio = "gmail.com"
    else:
        local = local.split("+", 1)[0]
    return f"{local}@{dominio}"


def contar_urls(texto: str) -> int:
    return len(RE_URL.findall(texto))


def validar_texto_libre(valor: str, minimo: int, maximo: int, multilinea: bool) -> str:
    valor = normalizar_texto(valor, multilinea=multilinea)
    if len(valor) < minimo:
        raise ValueError(f"Debe tener al menos {minimo} caracteres")
    if len(valor) > maximo:
        raise ValueError(f"Debe tener como máximo {maximo} caracteres")
    if RE_HTML.search(valor):
        raise ValueError("No se permite código HTML ni scripts")
    if contar_urls(valor) > 2:
        raise ValueError("Se permiten como máximo 2 enlaces")
    if RE_REPETIDO.search(valor):
        raise ValueError("El texto contiene caracteres repetidos en exceso")
    sin_espacios = re.sub(r"\s", "", valor)
    letras = sum(1 for c in sin_espacios if c.isalpha())
    if sin_espacios and letras / len(sin_espacios) < 0.55:
        raise ValueError("El texto debe contener principalmente letras")
    palabras = re.findall(r"\w+", valor.lower())
    if len(palabras) >= 6 and len(set(palabras)) / len(palabras) < 0.35:
        raise ValueError("El texto parece repetitivo")
    return valor


def calcular_puntaje_spam(asunto: str, descripcion: str, email: str) -> Tuple[int, List[str]]:
    texto = f"{asunto} {descripcion}".lower()
    puntaje = 0
    motivos: List[str] = []

    urls = contar_urls(texto)
    if urls >= 2:
        puntaje += 2
        motivos.append("varios_enlaces")
    elif urls == 1:
        puntaje += 1
        motivos.append("un_enlace")

    coincidencias = [p for p in PALABRAS_SPAM if p in texto]
    if coincidencias:
        puntaje += min(3, len(coincidencias) * 2)
        motivos.append("palabras_spam")

    letras = [c for c in f"{asunto} {descripcion}" if c.isalpha()]
    if len(letras) >= 20 and sum(1 for c in letras if c.isupper()) / len(letras) > 0.6:
        puntaje += 1
        motivos.append("mayusculas")

    if re.search(r"[!?$]{3,}", texto):
        puntaje += 1
        motivos.append("signos_excesivos")

    if len(re.findall(r"\+?\d[\d\s\-]{8,}\d", texto)) >= 2:
        puntaje += 1
        motivos.append("telefonos")

    local = email.split("@", 1)[0]
    if len(re.findall(r"\d", local)) >= 7:
        puntaje += 1
        motivos.append("correo_numerico")

    return puntaje, motivos


def _secreto() -> bytes:
    valor = os.getenv("CONTACTO_SECRETO") or os.getenv("SECRET_KEY")
    if valor:
        return valor.encode("utf-8")
    logger.warning("CONTACTO_SECRETO/SECRET_KEY no definido: se usa un secreto efímero por proceso")
    return _SECRETO_EFIMERO


_SECRETO_EFIMERO = secrets.token_bytes(32)


def _b64(datos: bytes) -> str:
    return base64.urlsafe_b64encode(datos).decode().rstrip("=")


def _desb64(texto: str) -> bytes:
    return base64.urlsafe_b64decode(texto + "=" * (-len(texto) % 4))


def emitir_token_formulario(ahora: Optional[float] = None) -> str:
    carga = {"t": int(ahora if ahora is not None else time.time()), "n": secrets.token_hex(6)}
    cuerpo = _b64(json.dumps(carga, separators=(",", ":")).encode())
    firma = hmac.new(_secreto(), cuerpo.encode(), hashlib.sha256).hexdigest()[:40]
    return f"{cuerpo}.{firma}"


class TokenInvalido(Exception):
    pass


class TokenExpirado(Exception):
    pass


class FormularioMuyRapido(Exception):
    pass


def validar_token_formulario(token: str, ahora: Optional[float] = None) -> int:
    ahora = ahora if ahora is not None else time.time()
    try:
        cuerpo, firma = token.split(".", 1)
        esperado = hmac.new(_secreto(), cuerpo.encode(), hashlib.sha256).hexdigest()[:40]
        if not hmac.compare_digest(firma, esperado):
            raise TokenInvalido()
        emitido = int(json.loads(_desb64(cuerpo))["t"])
    except TokenInvalido:
        raise
    except Exception:
        raise TokenInvalido()
    edad = ahora - emitido
    if edad < 0:
        raise TokenInvalido()
    if edad > TOKEN_VIGENCIA_SEGUNDOS:
        raise TokenExpirado()
    if edad < ESPERA_MINIMA_SEGUNDOS:
        raise FormularioMuyRapido()
    return int(edad)


def _pimienta() -> bytes:
    return (os.getenv("CONTACTO_PIMIENTA") or os.getenv("SECRET_KEY") or "").encode() or _secreto()


def hash_ip(ip: str) -> str:
    return hashlib.sha256(_pimienta() + b"|ip|" + ip.encode()).hexdigest()


def hash_contenido(email: str, asunto: str, descripcion: str) -> str:
    base = re.sub(r"\W+", " ", f"{asunto} {descripcion}".lower()).strip()
    return hashlib.sha256(_pimienta() + b"|msg|" + email_canonico(email).encode() + b"|" + base.encode()).hexdigest()


def obtener_ip_cliente(request) -> str:
    if os.getenv("CONTACTO_CONFIAR_PROXY", "true").lower() in ("1", "true", "yes"):
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()[:64]
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "desconocida")[:64]


def _contar(db: Session, columna, valor, desde: datetime) -> int:
    return (
        db.query(func.count(TicketSoporte.id))
        .filter(columna == valor, TicketSoporte.fecha_creacion >= desde)
        .scalar()
        or 0
    )


class LimiteExcedido(Exception):
    def __init__(self, detalle: str, reintentar_en: int):
        super().__init__(detalle)
        self.detalle = detalle
        self.reintentar_en = reintentar_en


def verificar_limites_publicos(db: Session, ip_h: str, email: str) -> None:
    ahora = datetime.utcnow()
    hora = ahora - timedelta(hours=1)
    dia = ahora - timedelta(days=1)

    if _contar(db, TicketSoporte.ip_hash, ip_h, hora) >= LIMITE_IP_HORA:
        raise LimiteExcedido("Demasiadas solicitudes desde tu conexión. Inténtalo más tarde.", 3600)
    if _contar(db, TicketSoporte.ip_hash, ip_h, dia) >= LIMITE_IP_DIA:
        raise LimiteExcedido("Se alcanzó el límite diario de solicitudes desde tu conexión.", 86400)
    if _contar(db, TicketSoporte.email, email, hora) >= LIMITE_EMAIL_HORA:
        raise LimiteExcedido("Este correo ya envió varias solicitudes. Inténtalo más tarde.", 3600)
    if _contar(db, TicketSoporte.email, email, dia) >= LIMITE_EMAIL_DIA:
        raise LimiteExcedido("Se alcanzó el límite diario de solicitudes para este correo.", 86400)


def verificar_limites_usuario(db: Session, usuario_id: int, ip_h: str) -> None:
    ahora = datetime.utcnow()
    dia = ahora - timedelta(days=1)
    hora = ahora - timedelta(hours=1)
    if _contar(db, TicketSoporte.usuario_id, usuario_id, dia) >= LIMITE_USUARIO_DIA:
        raise LimiteExcedido("Alcanzaste el límite diario de tickets.", 86400)
    if _contar(db, TicketSoporte.ip_hash, ip_h, hora) >= LIMITE_IP_HORA * 3:
        raise LimiteExcedido("Demasiadas solicitudes desde tu conexión. Inténtalo más tarde.", 3600)


def buscar_duplicado(db: Session, hash_c: str) -> Optional[TicketSoporte]:
    desde = datetime.utcnow() - timedelta(hours=VENTANA_DUPLICADOS_HORAS)
    return (
        db.query(TicketSoporte)
        .filter(TicketSoporte.hash_contenido == hash_c, TicketSoporte.fecha_creacion >= desde)
        .order_by(TicketSoporte.id.desc())
        .first()
    )


def turnstile_requerido() -> bool:
    return bool(os.getenv("TURNSTILE_SECRET"))


def verificar_turnstile(token: Optional[str], ip: str) -> bool:
    secreto = os.getenv("TURNSTILE_SECRET")
    if not secreto:
        return True
    if not token:
        return False
    try:
        import httpx

        respuesta = httpx.post(
            TURNSTILE_URL,
            data={"secret": secreto, "response": token, "remoteip": ip},
            timeout=5.0,
        )
        return bool(respuesta.json().get("success"))
    except Exception as e:
        logger.error("Fallo verificando Turnstile: %s", e)
        return False


class LimitadorMemoria:
    def __init__(self, maximo: int, ventana_segundos: int):
        self.maximo = maximo
        self.ventana = ventana_segundos
        self._eventos: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def permitir(self, clave: str, ahora: Optional[float] = None) -> bool:
        ahora = ahora if ahora is not None else time.monotonic()
        with self._lock:
            cola = self._eventos[clave]
            while cola and ahora - cola[0] > self.ventana:
                cola.popleft()
            if len(cola) >= self.maximo:
                return False
            cola.append(ahora)
            if len(self._eventos) > 10000:
                for k in [k for k, v in self._eventos.items() if not v]:
                    self._eventos.pop(k, None)
            return True

    def reiniciar(self) -> None:
        with self._lock:
            self._eventos.clear()
