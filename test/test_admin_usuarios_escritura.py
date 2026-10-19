"""
Pruebas de protección de las operaciones de ACTUALIZAR y ELIMINAR usuarios
del panel de administración (rutas PUT / PATCH / DELETE bajo /admin/usuarios).

Siguen el mismo estilo que test_rutas_admin.py: descubren las rutas desde
`app.routes` (así no hay que mantener una lista a mano) y usan la fixture
`client` del conftest.

Ninguna prueba puede modificar datos: todas las solicitudes se rechazan en la
capa de autenticación antes de llegar a la lógica de actualización/eliminación.
"""
from datetime import datetime, timedelta, timezone
import re

import pytest
from fastapi.routing import APIRoute
from jose import jwt

from app.main import app
from app.utilidades.configuracion import configuracion
from app.utilidades.seguridad import crear_access_token

METODOS_ESCRITURA = ("PUT", "PATCH", "DELETE")


def rutas_admin_usuarios_escritura():
    """Devuelve [(metodo, ruta)] de las rutas admin de usuarios que modifican datos."""
    resultado = set()
    for ruta in app.routes:
        if not isinstance(ruta, APIRoute):
            continue
        path = ruta.path.lower()
        if "admin" not in path or "usuarios" not in path:
            continue
        for metodo in METODOS_ESCRITURA:
            if metodo in ruta.methods:
                resultado.add((metodo, re.sub(r"\{[^}]+\}", "1", ruta.path)))
    return sorted(resultado)


RUTAS = rutas_admin_usuarios_escritura()
IDS = [f"{m} {r}" for m, r in RUTAS]


def test_existen_rutas_admin_de_actualizar_o_eliminar_usuarios():
    assert len(RUTAS) > 0, (
        "No se encontraron rutas PUT/PATCH/DELETE bajo /admin/usuarios; "
        "revisa el prefijo del router de administración de usuarios."
    )


@pytest.mark.parametrize("metodo,ruta", RUTAS, ids=IDS)
def test_modificar_usuario_sin_sesion_es_rechazado(client, metodo, ruta):
    res = client.request(metodo, ruta, json={}, follow_redirects=False)
    assert res.status_code in (401, 403), f"{metodo} {ruta} respondió {res.status_code}"


@pytest.mark.parametrize("metodo,ruta", RUTAS, ids=IDS)
def test_modificar_usuario_con_cookie_invalida_es_rechazado(client, metodo, ruta):
    res = client.request(
        metodo, ruta, json={},
        headers={"Cookie": "access_token=token.invalido.123"},
        follow_redirects=False,
    )
    assert res.status_code in (401, 403), f"{metodo} {ruta} respondió {res.status_code}"


@pytest.mark.parametrize("metodo,ruta", RUTAS, ids=IDS)
def test_modificar_usuario_con_token_de_otra_clave_es_rechazado(client, metodo, ruta):
    falso = jwt.encode(
        {
            "sub": "1",
            "sid": 1,
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "clave-de-atacante-" + "x" * 40,
        algorithm=configuracion.algorithm,
    )
    res = client.request(
        metodo, ruta, json={},
        headers={"Cookie": f"access_token={falso}"},
        follow_redirects=False,
    )
    assert res.status_code in (401, 403), f"{metodo} {ruta} respondió {res.status_code}"


@pytest.mark.parametrize("metodo,ruta", RUTAS, ids=IDS)
def test_modificar_usuario_con_sesion_inexistente_es_rechazado(client, metodo, ruta):
    token = crear_access_token(1, 999999999)
    res = client.request(
        metodo, ruta, json={},
        headers={"Cookie": f"access_token={token}"},
        follow_redirects=False,
    )
    assert res.status_code in (401, 403), f"{metodo} {ruta} respondió {res.status_code}"
