from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request, Response
from jose import jwt

from app.modelos.sesion_usuario import SesionUsuario
from app.modelos.usuario import Usuario
from app.utilidades import seguridad as seg
from app.utilidades.configuracion import configuracion


def hacer_request(metodo="GET", path="/", cookies=None, headers=None):
    lista = []
    if cookies:
        valor = "; ".join(f"{k}={v}" for k, v in cookies.items())
        lista.append((b"cookie", valor.encode()))
    for k, v in (headers or {}).items():
        lista.append((k.lower().encode(), v.encode()))
    scope = {
        "type": "http",
        "method": metodo,
        "path": path,
        "headers": lista,
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def firmar(payload, clave=None):
    return jwt.encode(
        payload,
        clave or configuracion.secret_key,
        algorithm=configuracion.algorithm,
    )


def payload_base(**extra):
    base = {
        "sub": "1",
        "sid": 1,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    base.update(extra)
    return base


def nueva_sesion(**kw):
    ahora = datetime.now(timezone.utc)
    base = dict(
        id=1,
        usuario_id=1,
        refresh_hash=seg._hash_token("actual"),
        refresh_hash_anterior=None,
        rotada_en=None,
        expira_en=ahora + timedelta(days=1),
        expira_absoluta_en=ahora + timedelta(days=30),
        revocada=False,
        revocada_en=None,
        revocada_motivo=None,
        ultimo_uso=ahora,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def bd_falsa(sesion, usuario=None):
    bd = MagicMock()
    bd.query.return_value.filter.return_value.first.return_value = sesion
    bd.get.return_value = usuario
    return bd


def usuario_falso(activo=True, rol="estudiante", permisos=()):
    return SimpleNamespace(
        id=1,
        activo=activo,
        rol=SimpleNamespace(codigo=rol),
        tiene_rol=lambda *codigos: rol in codigos,
        tiene_permiso=lambda codigo: codigo in permisos,
    )


def buscar_cookie(response, nombre):
    for h in response.headers.getlist("set-cookie"):
        if h.startswith(f"{nombre}="):
            return h
    return None


class TestPasswords:
    def test_hash_no_es_el_texto_plano(self):
        h = seg.obtener_hash_password("Clave123!")
        assert h != "Clave123!"
        assert h.startswith("$2")

    def test_verifica_password_correcta(self):
        h = seg.obtener_hash_password("Clave123!")
        assert seg.verificar_password("Clave123!", h) is True

    def test_rechaza_password_incorrecta(self):
        h = seg.obtener_hash_password("Clave123!")
        assert seg.verificar_password("otra", h) is False

    def test_mismo_password_genera_hashes_distintos(self):
        assert seg.obtener_hash_password("Clave123!") != seg.obtener_hash_password("Clave123!")

    def test_password_muy_larga_funciona(self):
        larga = "a" * 500
        h = seg.obtener_hash_password(larga)
        assert seg.verificar_password(larga, h) is True
        assert seg.verificar_password("a" * 499, h) is False

    def test_hash_invalido_devuelve_false(self):
        assert seg.verificar_password("x", "esto-no-es-un-hash") is False

    def test_verificacion_falsa_no_lanza_error(self):
        assert seg.verificar_password_falso("x") is None


class TestAccessToken:
    def test_contiene_los_datos_esperados(self):
        token = seg.crear_access_token(7, 9)
        data = jwt.decode(token, configuracion.secret_key, algorithms=[configuracion.algorithm])
        assert data["sub"] == "7"
        assert data["sid"] == 9
        assert data["type"] == "access"
        assert "jti" in data and "exp" in data and "iat" in data

    def test_cada_token_tiene_jti_unico(self):
        a = jwt.get_unverified_claims(seg.crear_access_token(1, 1))
        b = jwt.get_unverified_claims(seg.crear_access_token(1, 1))
        assert a["jti"] != b["jti"]

    def test_expiracion_personalizada(self):
        token = seg.crear_access_token(1, 1, expires_delta=timedelta(hours=2))
        data = jwt.get_unverified_claims(token)
        restante = data["exp"] - datetime.now(timezone.utc).timestamp()
        assert 7000 < restante <= 7200


class TestVerificarToken:
    def test_sin_cookie_es_401(self):
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(hacer_request())
        assert e.value.status_code == 401

    def test_token_basura_es_401(self):
        req = hacer_request(cookies={seg.COOKIE_ACCESS: "token.invalido.123"})
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(req)
        assert e.value.status_code == 401

    def test_firmado_con_otra_clave_es_401(self):
        token = firmar(payload_base(), clave="clave-de-atacante-" + "x" * 40)
        req = hacer_request(cookies={seg.COOKIE_ACCESS: token})
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(req)
        assert e.value.status_code == 401

    def test_token_expirado_es_401(self):
        token = seg.crear_access_token(1, 1, expires_delta=timedelta(seconds=-10))
        req = hacer_request(cookies={seg.COOKIE_ACCESS: token})
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(req)
        assert e.value.status_code == 401

    def test_tipo_distinto_de_access_es_401(self):
        token = firmar(payload_base(type="refresh"))
        req = hacer_request(cookies={seg.COOKIE_ACCESS: token})
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(req)
        assert e.value.status_code == 401

    def test_sin_sid_es_401(self):
        datos = payload_base()
        del datos["sid"]
        req = hacer_request(cookies={seg.COOKIE_ACCESS: firmar(datos)})
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(req)
        assert e.value.status_code == 401

    def test_sub_no_numerico_es_401(self):
        req = hacer_request(cookies={seg.COOKIE_ACCESS: firmar(payload_base(sub="abc"))})
        with pytest.raises(HTTPException) as e:
            seg.verificar_token(req)
        assert e.value.status_code == 401

    def test_token_valido_devuelve_payload(self):
        token = seg.crear_access_token(3, 8)
        req = hacer_request(cookies={seg.COOKIE_ACCESS: token})
        payload = seg.verificar_token(req)
        assert payload["_usuario_id"] == 3
        assert payload["_sesion_id"] == 8


class TestObtenerUsuarioActual:
    @pytest.fixture(autouse=True)
    def sin_auditoria(self, monkeypatch):
        monkeypatch.setattr(seg, "marcar_usuario_actual", lambda bd, uid: None)

    def bd(self, sesion, usuario):
        bd = MagicMock()

        def get(modelo, _id):
            if modelo is SesionUsuario:
                return sesion
            if modelo is Usuario:
                return usuario
            return None

        bd.get.side_effect = get
        return bd

    def payload(self, usuario_id=1, sesion_id=1):
        return {"_usuario_id": usuario_id, "_sesion_id": sesion_id}

    def test_sesion_inexistente_es_401(self):
        with pytest.raises(HTTPException) as e:
            seg.obtener_usuario_actual(hacer_request(), self.payload(), self.bd(None, usuario_falso()))
        assert e.value.status_code == 401

    def test_sesion_revocada_es_401(self):
        bd = self.bd(nueva_sesion(revocada=True), usuario_falso())
        with pytest.raises(HTTPException) as e:
            seg.obtener_usuario_actual(hacer_request(), self.payload(), bd)
        assert e.value.status_code == 401

    def test_sesion_de_otro_usuario_es_401(self):
        bd = self.bd(nueva_sesion(usuario_id=2), usuario_falso())
        with pytest.raises(HTTPException) as e:
            seg.obtener_usuario_actual(hacer_request(), self.payload(usuario_id=1), bd)
        assert e.value.status_code == 401

    def test_sesion_expirada_es_401(self):
        pasado = datetime.now(timezone.utc) - timedelta(minutes=1)
        bd = self.bd(nueva_sesion(expira_en=pasado), usuario_falso())
        with pytest.raises(HTTPException) as e:
            seg.obtener_usuario_actual(hacer_request(), self.payload(), bd)
        assert e.value.status_code == 401

    def test_usuario_inexistente_es_401(self):
        bd = self.bd(nueva_sesion(), None)
        with pytest.raises(HTTPException) as e:
            seg.obtener_usuario_actual(hacer_request(), self.payload(), bd)
        assert e.value.status_code == 401

    def test_usuario_inactivo_es_403(self):
        bd = self.bd(nueva_sesion(), usuario_falso(activo=False))
        with pytest.raises(HTTPException) as e:
            seg.obtener_usuario_actual(hacer_request(), self.payload(), bd)
        assert e.value.status_code == 403

    def test_sesion_valida_devuelve_usuario(self):
        usuario = usuario_falso()
        req = hacer_request()
        resultado = seg.obtener_usuario_actual(req, self.payload(), self.bd(nueva_sesion(), usuario))
        assert resultado is usuario
        assert req.state.sesion_id == 1


class TestRolesYPermisos:
    def test_rol_no_permitido_es_403(self):
        dependencia = seg.requiere_roles("admin")
        with pytest.raises(HTTPException) as e:
            dependencia(usuario_actual=usuario_falso(rol="estudiante"))
        assert e.value.status_code == 403

    def test_rol_permitido_pasa(self):
        usuario = usuario_falso(rol="admin")
        assert seg.requiere_roles("admin")(usuario_actual=usuario) is usuario

    def test_varios_roles_permitidos(self):
        usuario = usuario_falso(rol="docente")
        assert seg.requiere_roles("admin", "docente")(usuario_actual=usuario) is usuario

    def test_permiso_faltante_es_403(self):
        dependencia = seg.requiere_permiso("usuarios.eliminar")
        with pytest.raises(HTTPException) as e:
            dependencia(usuario_actual=usuario_falso(permisos=("usuarios.ver",)))
        assert e.value.status_code == 403

    def test_permiso_presente_pasa(self):
        usuario = usuario_falso(permisos=("usuarios.ver",))
        assert seg.requiere_permiso("usuarios.ver")(usuario_actual=usuario) is usuario

    def test_basta_uno_de_varios_permisos(self):
        usuario = usuario_falso(permisos=("b",))
        assert seg.requiere_permiso("a", "b")(usuario_actual=usuario) is usuario


class TestCsrf:
    RUTA = "/api/v1/usuarios/cambiar-password"

    def test_get_no_requiere_csrf(self):
        req = hacer_request("GET", self.RUTA, cookies={seg.COOKIE_ACCESS: "x"})
        assert seg.verificar_csrf(req) is None

    def test_post_sin_cookies_de_sesion_pasa(self):
        assert seg.verificar_csrf(hacer_request("POST", self.RUTA)) is None

    def test_post_con_sesion_y_sin_token_csrf_es_403(self):
        req = hacer_request("POST", self.RUTA, cookies={seg.COOKIE_ACCESS: "x"})
        with pytest.raises(HTTPException) as e:
            seg.verificar_csrf(req)
        assert e.value.status_code == 403

    def test_post_con_csrf_distinto_es_403(self):
        req = hacer_request(
            "POST",
            self.RUTA,
            cookies={seg.COOKIE_ACCESS: "x", seg.COOKIE_CSRF: "abc"},
            headers={seg.HEADER_CSRF: "xyz"},
        )
        with pytest.raises(HTTPException) as e:
            seg.verificar_csrf(req)
        assert e.value.status_code == 403

    def test_post_con_csrf_igual_pasa(self):
        req = hacer_request(
            "POST",
            self.RUTA,
            cookies={seg.COOKIE_ACCESS: "x", seg.COOKIE_CSRF: "abc"},
            headers={seg.HEADER_CSRF: "abc"},
        )
        assert seg.verificar_csrf(req) is None

    def test_origen_no_permitido_es_403(self):
        origen = "http://sitio-malicioso.example"
        assert origen not in configuracion.origenes_permitidos
        req = hacer_request(
            "POST",
            self.RUTA,
            cookies={seg.COOKIE_ACCESS: "x", seg.COOKIE_CSRF: "abc"},
            headers={seg.HEADER_CSRF: "abc", "origin": origen},
        )
        with pytest.raises(HTTPException) as e:
            seg.verificar_csrf(req)
        assert e.value.status_code == 403

    def test_origen_permitido_con_csrf_valido_pasa(self):
        permitidos = list(configuracion.origenes_permitidos)
        if not permitidos:
            pytest.skip("No hay orígenes permitidos configurados")
        req = hacer_request(
            "POST",
            self.RUTA,
            cookies={seg.COOKIE_ACCESS: "x", seg.COOKIE_CSRF: "abc"},
            headers={seg.HEADER_CSRF: "abc", "origin": permitidos[0]},
        )
        assert seg.verificar_csrf(req) is None

    @pytest.mark.parametrize("ruta", ["/api/v1/auth/login", "/api/v1/auth/registro"])
    def test_rutas_exentas_no_exigen_csrf(self, ruta):
        req = hacer_request("POST", ruta, cookies={seg.COOKIE_ACCESS: "x"})
        assert seg.verificar_csrf(req) is None


class TestCookies:
    def test_establece_las_tres_cookies_con_atributos_correctos(self):
        resp = Response()
        seg.establecer_cookies_sesion(resp, "acceso", "refresco")
        acceso = buscar_cookie(resp, seg.COOKIE_ACCESS)
        refresco = buscar_cookie(resp, seg.COOKIE_REFRESH)
        csrf = buscar_cookie(resp, seg.COOKIE_CSRF)
        assert acceso and "HttpOnly" in acceso
        assert refresco and "HttpOnly" in refresco
        assert f"Path={configuracion.cookie_refresh_path}" in refresco
        assert csrf and "HttpOnly" not in csrf

    def test_borrar_cookies_las_expira(self):
        resp = Response()
        seg.borrar_cookies_sesion(resp)
        for nombre in (seg.COOKIE_ACCESS, seg.COOKIE_REFRESH, seg.COOKIE_CSRF):
            cookie = buscar_cookie(resp, nombre)
            assert cookie is not None
            assert "Max-Age=0" in cookie


class TestCrearSesion:
    def test_guarda_solo_el_hash_del_refresh(self):
        bd = MagicMock()
        usuario = SimpleNamespace(id=5)
        req = hacer_request(headers={"user-agent": "pytest"})
        sesion, refresh = seg.crear_sesion(bd, usuario, req)
        assert sesion.usuario_id == 5
        assert sesion.refresh_hash == seg._hash_token(refresh)
        assert sesion.refresh_hash != refresh
        assert sesion.revocada is False
        assert sesion.expira_en <= sesion.expira_absoluta_en
        bd.add.assert_called_once()
        bd.commit.assert_called_once()

    def test_refresh_tokens_son_unicos(self):
        usuario = SimpleNamespace(id=5)
        _, a = seg.crear_sesion(MagicMock(), usuario, hacer_request())
        _, b = seg.crear_sesion(MagicMock(), usuario, hacer_request())
        assert a != b
        assert len(a) >= 48


class TestRotarSesion:
    def test_refresh_desconocido_es_401(self):
        with pytest.raises(HTTPException) as e:
            seg.rotar_sesion(bd_falsa(None), "cualquiera")
        assert e.value.status_code == 401

    def test_sesion_revocada_es_401(self):
        with pytest.raises(HTTPException) as e:
            seg.rotar_sesion(bd_falsa(nueva_sesion(revocada=True)), "actual")
        assert e.value.status_code == 401

    def test_rotacion_exitosa_entrega_nuevo_refresh(self):
        sesion = nueva_sesion()
        bd = bd_falsa(sesion, SimpleNamespace(id=1, activo=True))
        hash_previo = sesion.refresh_hash
        _, nuevo = seg.rotar_sesion(bd, "actual")
        assert nuevo != "actual"
        assert sesion.refresh_hash == seg._hash_token(nuevo)
        assert sesion.refresh_hash_anterior == hash_previo
        assert sesion.rotada_en is not None

    def test_reutilizar_refresh_viejo_fuera_de_gracia_revoca_la_sesion(self):
        ahora = datetime.now(timezone.utc)
        sesion = nueva_sesion(
            refresh_hash_anterior=seg._hash_token("viejo"),
            rotada_en=ahora - timedelta(seconds=120),
        )
        bd = bd_falsa(sesion, SimpleNamespace(id=1, activo=True))
        with pytest.raises(HTTPException) as e:
            seg.rotar_sesion(bd, "viejo")
        assert e.value.status_code == 401
        assert sesion.revocada is True
        assert sesion.revocada_motivo == "reutilizacion_refresh"

    def test_reutilizar_refresh_viejo_dentro_de_gracia_se_tolera(self):
        ahora = datetime.now(timezone.utc)
        sesion = nueva_sesion(
            refresh_hash_anterior=seg._hash_token("viejo"),
            rotada_en=ahora - timedelta(seconds=2),
        )
        bd = bd_falsa(sesion, SimpleNamespace(id=1, activo=True))
        _, nuevo = seg.rotar_sesion(bd, "viejo")
        assert nuevo
        assert sesion.revocada is False

    def test_sesion_expirada_se_revoca(self):
        pasado = datetime.now(timezone.utc) - timedelta(minutes=1)
        sesion = nueva_sesion(expira_en=pasado)
        bd = bd_falsa(sesion, SimpleNamespace(id=1, activo=True))
        with pytest.raises(HTTPException) as e:
            seg.rotar_sesion(bd, "actual")
        assert e.value.status_code == 401
        assert sesion.revocada_motivo == "expirada"

    def test_sesion_con_limite_absoluto_vencido_se_revoca(self):
        pasado = datetime.now(timezone.utc) - timedelta(minutes=1)
        sesion = nueva_sesion(expira_absoluta_en=pasado)
        bd = bd_falsa(sesion, SimpleNamespace(id=1, activo=True))
        with pytest.raises(HTTPException):
            seg.rotar_sesion(bd, "actual")
        assert sesion.revocada is True

    def test_usuario_inactivo_es_403_y_revoca(self):
        sesion = nueva_sesion()
        bd = bd_falsa(sesion, SimpleNamespace(id=1, activo=False))
        with pytest.raises(HTTPException) as e:
            seg.rotar_sesion(bd, "actual")
        assert e.value.status_code == 403
        assert sesion.revocada_motivo == "usuario_inactivo"


class TestRevocacion:
    def test_revoca_por_refresh(self):
        sesion = nueva_sesion()
        seg.revocar_sesion_por_refresh(bd_falsa(sesion), "actual", "logout")
        assert sesion.revocada is True
        assert sesion.revocada_motivo == "logout"

    def test_refresh_desconocido_no_falla(self):
        assert seg.revocar_sesion_por_refresh(bd_falsa(None), "x") is None

    def test_no_vuelve_a_revocar_una_sesion_ya_revocada(self):
        sesion = nueva_sesion(revocada=True, revocada_motivo="original")
        seg.revocar_sesion_por_refresh(bd_falsa(sesion), "actual", "logout")
        assert sesion.revocada_motivo == "original"


class TestUtilidadesInternas:
    def test_aware_agrega_zona_horaria(self):
        ingenuo = datetime(2026, 1, 1, 12, 0, 0)
        assert seg._aware(ingenuo).tzinfo == timezone.utc

    def test_aware_respeta_zona_existente(self):
        con_zona = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert seg._aware(con_zona) is con_zona

    def test_hash_token_es_determinista_y_no_reversible(self):
        assert seg._hash_token("abc") == seg._hash_token("abc")
        assert seg._hash_token("abc") != seg._hash_token("abd")
        assert len(seg._hash_token("abc")) == 64