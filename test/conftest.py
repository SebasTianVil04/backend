import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import app
from app.utilidades.base_datos import engine_owner

PASSWORD = "Prueba123!xyz"


@pytest.fixture(scope="session")
def client():
    with TestClient(app, base_url="https://testserver") as c:
        yield c


@pytest.fixture(scope="session")
def owner_scalar():
    def _consulta(sql, **params):
        with engine_owner.connect() as conn:
            return conn.execute(text(sql), params).scalar()
    return _consulta


@pytest.fixture(scope="session")
def owner_exec():
    def _ejecutar(sql, **params):
        with engine_owner.begin() as conn:
            conn.execute(text(sql), params)
    return _ejecutar


def _registrar():
    sufijo = uuid.uuid4().hex[:10]
    cliente = TestClient(app, base_url="https://testserver")
    respuesta = cliente.post(
        "/api/v1/auth/registro",
        json={
            "nombre_usuario": f"test_{sufijo}",
            "email": f"test_{sufijo}@example.com",
            "password": PASSWORD,
        },
    )
    assert respuesta.status_code == 200, respuesta.text
    return cliente, respuesta.json()["usuario"]["id"]


@pytest.fixture(scope="session")
def usuario_a():
    cliente, uid = _registrar()
    yield {"cliente": cliente, "id": uid}


@pytest.fixture(scope="session")
def usuario_b():
    cliente, uid = _registrar()
    yield {"cliente": cliente, "id": uid}


@pytest.fixture(scope="session")
def admin(owner_exec):
    cliente, uid = _registrar()
    owner_exec(
        "UPDATE usuarios SET rol_id = (SELECT id FROM roles WHERE codigo = 'admin') WHERE id = :id",
        id=uid,
    )
    yield {"cliente": cliente, "id": uid}


@pytest.fixture(scope="session", autouse=True)
def limpiar_usuarios_de_prueba():
    yield
    with engine_owner.begin() as conn:
        conn.execute(text(
            "DELETE FROM sesiones_usuario WHERE usuario_id IN "
            "(SELECT id FROM usuarios WHERE email LIKE 'test\\_%@example.com')"
        ))
        conn.execute(text("DELETE FROM usuarios WHERE email LIKE 'test\\_%@example.com'"))