from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import Response
from jose import jwt

from app.utilidades import seguridad as seg
from app.utilidades.configuracion import configuracion


class TestRevocarTodasLasSesiones:
    def test_revoca_todas_las_sesiones_activas_del_usuario(self):
        bd = MagicMock()
        consulta = bd.query.return_value.filter.return_value
        seg.revocar_todas_las_sesiones(bd, 5, "cambio_password")
        consulta.update.assert_called_once()
        valores = consulta.update.call_args.args[0]
        assert valores["revocada"] is True
        assert valores["revocada_motivo"] == "cambio_password"
        assert valores["revocada_en"] is not None
        assert consulta.update.call_args.kwargs == {"synchronize_session": False}
        bd.commit.assert_called_once()

    def test_sin_excepcion_no_aplica_filtro_adicional(self):
        bd = MagicMock()
        consulta = bd.query.return_value.filter.return_value
        seg.revocar_todas_las_sesiones(bd, 5, "logout_global")
        consulta.filter.assert_not_called()

    def test_con_excepcion_excluye_la_sesion_indicada(self):
        bd = MagicMock()
        consulta = bd.query.return_value.filter.return_value
        filtrada = consulta.filter.return_value
        seg.revocar_todas_las_sesiones(bd, 5, "cambio_password", excepto_sesion_id=9)
        consulta.filter.assert_called_once()
        filtrada.update.assert_called_once()
        consulta.update.assert_not_called()
        bd.commit.assert_called_once()


class TestIniciarSesion:
    def test_crea_sesion_y_deja_las_tres_cookies(self, monkeypatch):
        sesion = SimpleNamespace(id=4)
        monkeypatch.setattr(seg, "crear_sesion", lambda bd, usuario, request: (sesion, "refresh-plano"))
        response = Response()
        resultado = seg.iniciar_sesion(MagicMock(), SimpleNamespace(id=7), object(), response)
        assert resultado is sesion
        cookies = response.headers.getlist("set-cookie")
        nombres = {c.split("=", 1)[0] for c in cookies}
        assert {seg.COOKIE_ACCESS, seg.COOKIE_REFRESH, seg.COOKIE_CSRF} <= nombres

    def test_el_access_token_apunta_al_usuario_y_a_la_sesion(self, monkeypatch):
        monkeypatch.setattr(
            seg, "crear_sesion", lambda bd, usuario, request: (SimpleNamespace(id=4), "refresh-plano")
        )
        response = Response()
        seg.iniciar_sesion(MagicMock(), SimpleNamespace(id=7), object(), response)
        cookie = next(c for c in response.headers.getlist("set-cookie") if c.startswith(f"{seg.COOKIE_ACCESS}="))
        token = cookie.split("=", 1)[1].split(";", 1)[0]
        datos = jwt.decode(token, configuracion.secret_key, algorithms=[configuracion.algorithm])
        assert datos["sub"] == "7"
        assert datos["sid"] == 4

    def test_el_refresh_de_la_cookie_es_el_generado_por_la_sesion(self, monkeypatch):
        monkeypatch.setattr(
            seg, "crear_sesion", lambda bd, usuario, request: (SimpleNamespace(id=4), "refresh-plano")
        )
        response = Response()
        seg.iniciar_sesion(MagicMock(), SimpleNamespace(id=7), object(), response)
        cookie = next(c for c in response.headers.getlist("set-cookie") if c.startswith(f"{seg.COOKIE_REFRESH}="))
        assert cookie.startswith(f"{seg.COOKIE_REFRESH}=refresh-plano")