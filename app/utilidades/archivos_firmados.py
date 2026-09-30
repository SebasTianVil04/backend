import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Optional, Tuple

from .configuracion import configuracion

EXPIRACION_SEGUNDOS = 3600

EXTENSIONES_SERVIBLES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp4", ".webm", ".mov", ".avi",
}

RAICES = {
    "senas_referencia": {
        "ruta": (Path("archivos_subidos") / "senas_referencia").resolve(),
        "permiso": None,
        "expiracion": 3600,
    },
    "videos_dataset": {
        "ruta": (Path("archivos_subidos") / "videos_dataset").resolve(),
        "permiso": "dataset.ver",
        "expiracion": 300,
    },
    "videos_lecciones": {
        "ruta": (Path(configuracion.upload_dir) / "videos_lecciones").resolve(),
        "permiso": None,
        "expiracion": 3600,
    },
    "avatares": {
        "ruta": (Path(configuracion.upload_dir) / "avatares_usuarios").resolve(),
        "permiso": None,
        "expiracion": 3600,
    },
}


def _clave() -> bytes:
    return hashlib.sha256(f"{configuracion.secret_key}:archivos-firmados".encode("utf-8")).digest()


def _b64(datos: bytes) -> str:
    return base64.urlsafe_b64encode(datos).rstrip(b"=").decode("ascii")


def _desb64(texto: str) -> bytes:
    relleno = "=" * (-len(texto) % 4)
    return base64.urlsafe_b64decode(texto + relleno)


def permiso_requerido(raiz: str) -> Optional[str]:
    return RAICES[raiz]["permiso"]


def expiracion_de(raiz: str) -> int:
    config = RAICES.get(raiz)
    if not config:
        return EXPIRACION_SEGUNDOS
    return int(config.get("expiracion", EXPIRACION_SEGUNDOS))


def resolver_ruta(raiz: str, ruta: str) -> Optional[Path]:
    config = RAICES.get(raiz)
    if not config or not ruta or "\x00" in ruta:
        return None
    try:
        base = config["ruta"]
        candidato = (base / ruta).resolve()
        if not candidato.is_relative_to(base):
            return None
        if candidato.suffix.lower() not in EXTENSIONES_SERVIBLES:
            return None
        if not candidato.is_file():
            return None
        return candidato
    except (ValueError, OSError):
        return None


def generar_token(raiz: str, ruta: str, segundos: Optional[int] = None) -> str:
    vigencia = expiracion_de(raiz) if segundos is None else segundos
    payload = json.dumps(
        {"r": raiz, "p": ruta, "e": int(time.time()) + vigencia},
        separators=(",", ":"),
    ).encode("utf-8")
    cuerpo = _b64(payload)
    firma = _b64(hmac.new(_clave(), cuerpo.encode("ascii"), hashlib.sha256).digest())
    return f"{cuerpo}.{firma}"


def verificar_token(token: str) -> Optional[Tuple[str, str]]:
    try:
        cuerpo, firma = token.split(".", 1)
        esperada = _b64(hmac.new(_clave(), cuerpo.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(firma, esperada):
            return None
        datos = json.loads(_desb64(cuerpo))
        if int(datos["e"]) < int(time.time()):
            return None
        raiz = str(datos["r"])
        if raiz not in RAICES:
            return None
        return raiz, str(datos["p"])
    except Exception:
        return None