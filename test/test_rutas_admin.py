import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.routing import APIRoute
from jose import jwt

from app.main import app
from app.utilidades.configuracion import configuracion
from app.utilidades.seguridad import crear_access_token


def rutas_admin_get():
    resultado = []
    for ruta in app.routes:
        if not isinstance(ruta, APIRoute):
            continue
        if "admin" not in ruta.path.lower():
            continue
        if "GET" not in ruta.methods:
            continue
        ruta_formateada = re.sub(r"\{[^}]+\}", "1", ruta.path)
        resultado.append(ruta_formateada)
    return sorted(set(resultado))


@pytest.mark.parametrize("ruta", rutas_admin_get())
def test_ruta_admin_get_requiere_autenticacion(client, ruta):
    res = client.get(ruta, follow_redirects=False)
    assert res.status_code in (401, 403), f"{ruta} respondió {res.status_code}"


def test_existen_rutas_admin_registradas():
    assert len(rutas_admin_get()) > 0


def test_cookie_con_token_invalido_es_rechazada(client):
    res = client.get(
        "/api/v1/usuarios/perfil",
        headers={"Cookie": "access_token=token.invalido.123"},
    )
    assert res.status_code == 401


def test_token_firmado_con_otra_clave_es_rechazado(client):
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
    res = client.get("/api/v1/admin/usuarios", headers={"Cookie": f"access_token={falso}"})
    assert res.status_code == 401


def test_token_bien_firmado_con_sesion_inexistente_es_rechazado(client):
    token = crear_access_token(1, 999999999)
    res = client.get("/api/v1/admin/usuarios", headers={"Cookie": f"access_token={token}"})
    assert res.status_code == 401