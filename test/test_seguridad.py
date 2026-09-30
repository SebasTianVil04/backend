import pytest

RUTAS_SENSIBLES = ["/uploads", "/archivos_subidos", "/archivos/senas_referencia"]


@pytest.mark.parametrize("ruta", RUTAS_SENSIBLES)
def test_ruta_sensible_no_es_publica(client, ruta):
    res = client.get(ruta, follow_redirects=False)
    assert res.status_code in (401, 403, 404)


@pytest.mark.parametrize("ruta", RUTAS_SENSIBLES)
def test_ruta_sensible_no_lista_directorio(client, ruta):
    res = client.get(f"{ruta}/", follow_redirects=False)
    assert res.status_code != 200
    assert "index of" not in res.text.lower()


def test_perfil_requiere_token(client):
    res = client.get("/api/v1/usuarios/perfil")
    assert res.status_code == 401


def test_refresh_sin_cookie_es_rechazado(client):
    res = client.post("/api/v1/auth/refresh")
    assert res.status_code in (401, 403)