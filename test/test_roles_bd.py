from contextlib import contextmanager

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.utilidades.base_datos import engine

TABLAS_RLS = [
    "usuarios", "practicas", "progreso_lecciones", "progreso_clases",
    "resultados_examenes", "entrenamientos", "calibraciones_usuario",
    "sesiones_estudio", "videos_dataset",
]

ROLES_BD = ["rol_bd_auth", "rol_bd_usuario", "rol_bd_admin", "rol_bd_lectura"]


@contextmanager
def como(rol, uid=None):
    with engine.connect() as conn:
        trans = conn.begin()
        conn.execute(
            text("SELECT set_config('role', :r, true), set_config('app.usuario_id', :u, true)"),
            {"r": rol, "u": "" if uid is None else str(uid)},
        )
        try:
            yield conn
        finally:
            trans.rollback()


def csrf(cliente):
    return {"X-CSRF-Token": cliente.cookies.get("csrf_token")}


@pytest.mark.parametrize("rol", ["signafree_owner", "signafree_app"] + ROLES_BD)
def test_roles_sin_atributos_peligrosos(owner_scalar, rol):
    assert owner_scalar("SELECT count(*) FROM pg_roles WHERE rolname = :r", r=rol) == 1
    assert owner_scalar("SELECT rolsuper FROM pg_roles WHERE rolname = :r", r=rol) is False
    assert owner_scalar("SELECT rolbypassrls FROM pg_roles WHERE rolname = :r", r=rol) is False
    assert owner_scalar("SELECT rolcreaterole FROM pg_roles WHERE rolname = :r", r=rol) is False
    assert owner_scalar("SELECT rolcreatedb FROM pg_roles WHERE rolname = :r", r=rol) is False


def test_signafree_app_login_y_noinherit(owner_scalar):
    assert owner_scalar("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'signafree_app'") is True
    assert owner_scalar("SELECT rolinherit FROM pg_roles WHERE rolname = 'signafree_app'") is False


@pytest.mark.parametrize("rol", ROLES_BD)
def test_roles_bd_no_login_y_app_es_miembro(owner_scalar, rol):
    assert owner_scalar("SELECT rolcanlogin FROM pg_roles WHERE rolname = :r", r=rol) is False
    assert owner_scalar("SELECT pg_has_role('signafree_app', :r, 'MEMBER')", r=rol) is True


@pytest.mark.parametrize("tabla", ["usuarios", "sesiones_usuario", "roles", "practicas"])
def test_signafree_app_sin_privilegios_directos(owner_scalar, tabla):
    assert owner_scalar(
        "SELECT has_table_privilege('signafree_app', :t, 'SELECT,INSERT,UPDATE,DELETE')", t=tabla
    ) is False


@pytest.mark.parametrize("tabla", TABLAS_RLS)
def test_rls_activo(owner_scalar, tabla):
    assert owner_scalar(
        "SELECT rowsecurity FROM pg_tables WHERE schemaname = 'public' AND tablename = :t", t=tabla
    ) is True


@pytest.mark.parametrize(
    "rol,tabla,privilegio,esperado",
    [
        ("rol_bd_usuario", "roles", "SELECT", True),
        ("rol_bd_usuario", "roles", "INSERT", False),
        ("rol_bd_usuario", "permisos", "UPDATE", False),
        ("rol_bd_usuario", "rol_permisos", "DELETE", False),
        ("rol_bd_usuario", "sesiones_usuario", "SELECT", False),
        ("rol_bd_usuario", "tokens_recuperacion", "SELECT", False),
        ("rol_bd_usuario", "usuarios", "DELETE", False),
        # Se eliminó la fila ("rol_bd_usuario", "usuarios", "UPDATE", True) porque
        # ahora el UPDATE es por columna y has_table_privilege devuelve False.
        # Se verifica en test_usuario_update_usuarios_por_columna.
        ("rol_bd_usuario", "practicas", "INSERT", True),
        ("rol_bd_usuario", "practicas", "DELETE", False),
        ("rol_bd_auth", "sesiones_usuario", "DELETE", True),
        ("rol_bd_auth", "tokens_recuperacion", "INSERT", True),
        ("rol_bd_auth", "practicas", "SELECT", False),
        ("rol_bd_lectura", "sesiones_usuario", "SELECT", False),
        ("rol_bd_lectura", "tokens_recuperacion", "SELECT", False),
        ("rol_bd_lectura", "practicas", "SELECT", True),
        ("rol_bd_lectura", "practicas", "INSERT", False),
        ("rol_bd_admin", "usuarios", "DELETE", True),
    ],
)
def test_matriz_privilegios(owner_scalar, rol, tabla, privilegio, esperado):
    assert owner_scalar(
        "SELECT has_table_privilege(:r, :t, :p)", r=rol, t=tabla, p=privilegio
    ) is esperado


def test_usuario_update_usuarios_por_columna(owner_scalar):
    """rol_bd_usuario tiene UPDATE por columna sobre usuarios, no sobre toda la tabla."""
    # Columnas editables: sí
    for col in ("nombre_usuario", "email", "password_hash", "fecha_actualizacion"):
        assert owner_scalar(
            "SELECT has_column_privilege('rol_bd_usuario', 'usuarios', :c, 'UPDATE')", c=col
        ) is True

    # Columnas sensibles: no
    for col in ("rol_id", "activo", "verificado", "creado_por_id", "actualizado_por_id"):
        assert owner_scalar(
            "SELECT has_column_privilege('rol_bd_usuario', 'usuarios', :c, 'UPDATE')", c=col
        ) is False

    # A nivel de tabla, has_table_privilege devuelve False porque no es UPDATE sobre todas las columnas
    assert owner_scalar(
        "SELECT has_table_privilege('rol_bd_usuario', 'usuarios', 'UPDATE')"
    ) is False


def test_lectura_no_ve_password_hash(owner_scalar):
    assert owner_scalar(
        "SELECT has_column_privilege('rol_bd_lectura', 'usuarios', 'password_hash', 'SELECT')"
    ) is False
    assert owner_scalar(
        "SELECT has_column_privilege('rol_bd_lectura', 'usuarios', 'email', 'SELECT')"
    ) is True


def test_usuario_no_lee_password_hash(owner_scalar):
    """rol_bd_usuario no debe poder leer password_hash de usuarios."""
    assert owner_scalar(
        "SELECT has_column_privilege('rol_bd_usuario', 'usuarios', 'password_hash', 'SELECT')"
    ) is False
    assert owner_scalar(
        "SELECT has_column_privilege('rol_bd_usuario', 'usuarios', 'email', 'SELECT')"
    ) is True


def test_usuario_no_puede_select_password_hash(usuario_a):
    """Verifica en runtime que un SELECT real sobre password_hash falla."""
    with pytest.raises(DBAPIError):
        with como("rol_bd_usuario", usuario_a["id"]) as conn:
            conn.execute(text("SELECT password_hash FROM usuarios LIMIT 1"))


@pytest.mark.parametrize("rol", ["rol_bd_usuario", "rol_bd_auth", "rol_bd_lectura"])
def test_solo_admin_ejecuta_sincronizacion(owner_scalar, rol):
    assert owner_scalar(
        "SELECT has_function_privilege(:r, 'public.sincronizar_rol_bd(text)', 'EXECUTE')", r=rol
    ) is False
    assert owner_scalar(
        "SELECT has_function_privilege('rol_bd_admin', 'public.sincronizar_rol_bd(text)', 'EXECUTE')"
    ) is True


def test_rls_usuario_solo_actualiza_su_fila(usuario_a, usuario_b):
    with como("rol_bd_usuario", usuario_a["id"]) as conn:
        propia = conn.execute(
            text("UPDATE usuarios SET nombre_usuario = nombre_usuario WHERE id = :id"),
            {"id": usuario_a["id"]},
        ).rowcount
        ajena = conn.execute(
            text("UPDATE usuarios SET nombre_usuario = nombre_usuario WHERE id = :id"),
            {"id": usuario_b["id"]},
        ).rowcount
    assert propia == 1
    assert ajena == 0


def test_rls_sin_contexto_no_actualiza_nada(usuario_a):
    with como("rol_bd_usuario", None) as conn:
        filas = conn.execute(
            text("UPDATE usuarios SET nombre_usuario = nombre_usuario WHERE id = :id"),
            {"id": usuario_a["id"]},
        ).rowcount
    assert filas == 0


def test_rls_usuario_no_puede_borrar_usuarios(usuario_a, usuario_b):
    with pytest.raises(DBAPIError):
        with como("rol_bd_usuario", usuario_a["id"]) as conn:
            conn.execute(text("DELETE FROM usuarios WHERE id = :id"), {"id": usuario_b["id"]})


def test_rls_usuario_no_lee_sesiones(usuario_a):
    with pytest.raises(DBAPIError):
        with como("rol_bd_usuario", usuario_a["id"]) as conn:
            conn.execute(text("SELECT * FROM sesiones_usuario"))


def test_rls_lectura_no_lee_password_hash(usuario_a):
    with pytest.raises(DBAPIError):
        with como("rol_bd_lectura", usuario_a["id"]) as conn:
            conn.execute(text("SELECT password_hash FROM usuarios"))
    with como("rol_bd_lectura", usuario_a["id"]) as conn:
        assert conn.execute(text("SELECT count(*) FROM usuarios")).scalar() >= 1


def test_rls_auth_no_toca_practicas():
    with pytest.raises(DBAPIError):
        with como("rol_bd_auth") as conn:
            conn.execute(text("SELECT * FROM practicas"))


def test_usuario_normal_no_gestiona_roles(usuario_a):
    respuesta = usuario_a["cliente"].get("/api/v1/roles")
    assert respuesta.status_code == 403


def test_admin_lista_roles(admin):
    respuesta = admin["cliente"].get("/api/v1/roles")
    assert respuesta.status_code == 200


def test_ciclo_rol_personalizado(admin, owner_scalar):
    cliente = admin["cliente"]
    codigo = "test_rls_rol"
    rol_bd = f"rol_app_{codigo}"

    permisos = {p["codigo"]: p["id"] for p in cliente.get("/api/v1/roles/permisos").json()}
    assert "clases.crear" in permisos and "clases.eliminar" in permisos

    creado = cliente.post(
        "/api/v1/roles",
        json={
            "codigo": codigo,
            "nombre": "Test RLS",
            "descripcion": "rol de prueba",
            "permisos_ids": [permisos["clases.crear"]],
        },
        headers=csrf(cliente),
    )
    assert creado.status_code == 201, creado.text
    rol_id = creado.json()["id"]

    try:
        assert owner_scalar("SELECT count(*) FROM pg_roles WHERE rolname = :r", r=rol_bd) == 1
        assert owner_scalar("SELECT pg_has_role('signafree_app', :r, 'MEMBER')", r=rol_bd) is True
        assert owner_scalar("SELECT rolbypassrls FROM pg_roles WHERE rolname = :r", r=rol_bd) is False
        assert owner_scalar("SELECT has_table_privilege(:r, 'clases', 'INSERT')", r=rol_bd) is True
        assert owner_scalar("SELECT has_table_privilege(:r, 'clases', 'DELETE')", r=rol_bd) is False
        assert owner_scalar("SELECT has_table_privilege(:r, 'sesiones_usuario', 'SELECT')", r=rol_bd) is False
        assert owner_scalar("SELECT has_table_privilege(:r, 'usuarios', 'DELETE')", r=rol_bd) is False
        # Los roles personalizados heredan de rol_bd_usuario y, por tanto,
        # NO pueden leer password_hash (column-level grant).
        assert owner_scalar(
            "SELECT has_column_privilege(:r, 'usuarios', 'password_hash', 'SELECT')", r=rol_bd
        ) is False

        cambio = cliente.put(
            f"/api/v1/roles/{rol_id}/permisos",
            json={"permisos_ids": [permisos["clases.eliminar"]]},
            headers=csrf(cliente),
        )
        assert cambio.status_code == 200, cambio.text
        assert owner_scalar("SELECT has_table_privilege(:r, 'clases', 'INSERT')", r=rol_bd) is False
        assert owner_scalar("SELECT has_table_privilege(:r, 'clases', 'DELETE')", r=rol_bd) is True
    finally:
        borrado = cliente.delete(f"/api/v1/roles/{rol_id}", headers=csrf(cliente))
        assert borrado.status_code == 204, borrado.text

    assert owner_scalar("SELECT count(*) FROM pg_roles WHERE rolname = :r", r=rol_bd) == 0