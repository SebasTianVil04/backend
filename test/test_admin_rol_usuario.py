import os
import re
import uuid

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.main import app

METODOS = ("GET", "POST", "PUT", "PATCH", "DELETE")
ID_INEXISTENTE = "999999999"


def rutas_admin():
    resultado = set()
    for ruta in app.routes:
        if not isinstance(ruta, APIRoute):
            continue
        if "/admin" not in ruta.path.lower():
            continue
        for metodo in METODOS:
            if metodo in ruta.methods:
                resultado.add((metodo, re.sub(r"\{[^}]+\}", ID_INEXISTENTE, ruta.path)))
    return sorted(resultado)


RUTAS = rutas_admin()
IDS = [f"{m} {r}" for m, r in RUTAS]


def _url_owner() -> str:
    """
    URL con privilegios de owner (signafree_owner) para tareas de limpieza.
    No usamos la sesión de la app porque signafree_app ya no tiene DELETE
    sobre `usuarios` (lo cual es el comportamiento esperado).
    """
    url = os.environ.get("direct_url")
    if not url:
        raise RuntimeError(
            "Falta la variable de entorno 'direct_url' (necesaria para limpiar "
            "los usuarios de prueba). Debería venir de .env.backend."
        )
    return url


@pytest.fixture(scope="module")
def cliente_usuario():
    nombre = "qa_" + uuid.uuid4().hex[:8]
    email = nombre + "@correo.com"

    with TestClient(app, base_url="https://testserver") as c:
        res = c.post(
            "/api/v1/auth/registro",
            json={"nombre_usuario": nombre, "email": email, "password": "Segura#2026"},
        )
        assert res.status_code in (200, 201), res.text
        yield c

    # Cleanup con el usuario owner (signafree_owner).
    # No usamos obtener_bd() porque esa sesión va como signafree_app y
    # ya no tiene privilegio DELETE sobre `usuarios`.
    engine = create_engine(_url_owner(), future=True)
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM usuarios WHERE email = :e"), {"e": email})
    finally:
        engine.dispose()


def test_existen_rutas_admin():
    assert len(RUTAS) > 0


def test_usuario_autenticado_tiene_sesion_valida(cliente_usuario):
    res = cliente_usuario.get("/api/v1/categorias")
    assert res.status_code == 200, res.text


@pytest.mark.parametrize("metodo,ruta", RUTAS, ids=IDS)
def test_usuario_sin_rol_admin_recibe_403(cliente_usuario, metodo, ruta):
    headers = {}
    if metodo != "GET":
        headers["X-CSRF-Token"] = cliente_usuario.cookies.get("csrf_token") or ""
    res = cliente_usuario.request(
        metodo,
        ruta,
        json={} if metodo != "GET" else None,
        headers=headers,
        follow_redirects=False,
    )
    assert res.status_code == 403, f"{metodo} {ruta} respondio {res.status_code}"