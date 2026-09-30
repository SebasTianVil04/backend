import mimetypes
import re
from pathlib import Path
from typing import Dict, Iterator, Optional

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

TAMANO_BLOQUE = 1024 * 1024

_PATRON_RANGO = re.compile(r"^bytes=(\d*)-(\d*)$")


class _RangoNoSatisfacible(Exception):
    pass


def _leer_bloques(ruta: Path, inicio: int, fin: int) -> Iterator[bytes]:
    restante = fin - inicio + 1
    if restante <= 0:
        return
    with open(ruta, "rb") as archivo:
        archivo.seek(inicio)
        while restante > 0:
            bloque = archivo.read(min(TAMANO_BLOQUE, restante))
            if not bloque:
                break
            restante -= len(bloque)
            yield bloque


def _interpretar_rango(cabecera: str, tamano: int) -> Optional[tuple[int, int]]:
    coincidencia = _PATRON_RANGO.match(cabecera.strip())
    if not coincidencia:
        return None

    texto_inicio, texto_fin = coincidencia.groups()

    if texto_inicio == "" and texto_fin == "":
        return None

    if texto_inicio == "":
        sufijo = int(texto_fin)
        if sufijo == 0:
            raise _RangoNoSatisfacible()
        inicio = max(0, tamano - sufijo)
        fin = tamano - 1
    else:
        inicio = int(texto_inicio)
        fin = int(texto_fin) if texto_fin else tamano - 1
        fin = min(fin, tamano - 1)

    if tamano == 0 or inicio >= tamano or inicio > fin:
        raise _RangoNoSatisfacible()

    return inicio, fin


def archivo_con_rangos(
    request: Request,
    ruta: Path,
    tipo_contenido: Optional[str] = None,
    cache_control: str = "private, max-age=3600",
    cabeceras_extra: Optional[Dict[str, str]] = None,
) -> Response:
    tamano = ruta.stat().st_size
    tipo = tipo_contenido or mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"

    cabeceras = {
        "Accept-Ranges": "bytes",
        "Cache-Control": cache_control,
        **(cabeceras_extra or {}),
    }

    cabecera_rango = request.headers.get("range")
    if cabecera_rango:
        try:
            rango = _interpretar_rango(cabecera_rango, tamano)
        except _RangoNoSatisfacible:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{tamano}", "Accept-Ranges": "bytes"},
            )

        if rango is not None:
            inicio, fin = rango
            return StreamingResponse(
                _leer_bloques(ruta, inicio, fin),
                status_code=206,
                media_type=tipo,
                headers={
                    **cabeceras,
                    "Content-Range": f"bytes {inicio}-{fin}/{tamano}",
                    "Content-Length": str(fin - inicio + 1),
                },
            )

    return StreamingResponse(
        _leer_bloques(ruta, 0, tamano - 1),
        media_type=tipo,
        headers={**cabeceras, "Content-Length": str(tamano)},
    )