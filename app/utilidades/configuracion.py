import json
import base64
from typing import Any, Dict, List, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PREFIJOS_CLAVE_INSEGURA = ("tu_clave", "cambiar", "changeme", "secret", "password")
ALGORITMOS_PERMITIDOS = {"HS256", "HS384", "HS512"}
BCRYPT_RONDAS_MINIMAS = 10


class Configuracion(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    database_url: str
    direct_url: Optional[str] = None
    rls_activo: bool = True

    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    entorno: str = "dev"
    secret_key: str
    algorithm: str = "HS256"

    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    sesion_max_dias: int = 30
    reset_token_expire_minutes: int = 30

    bcrypt_rounds: int = 12

    cookie_secure: bool = True
    cookie_samesite: str = "lax"
    cookie_refresh_path: str = "/api/v1/auth"
    cookie_domain_csrf: Optional[str] = None

    origenes_permitidos: List[str] = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sebastianvil04.github.io",
        "http://38.56.216.65",
        "http://38.56.216.65:4200",
        "https://38.56.216.65",
    ]

    max_file_size: int = 10 * 1024 * 1024
    upload_dir: str = "archivos_subidos"
    temp_dir: str = "archivos_subidos/temp"

    usar_google_drive: bool = True
    google_drive_carpeta_id: str = ""
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    google_refresh_token: Optional[str] = None
    google_token_uri: str = "https://oauth2.googleapis.com/token"
    google_service_account_json_b64: Optional[str] = None

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    mail_from: str = "SignaFree <noreply@signafree.com>"

    frontend_url: str = "http://localhost:4200"

    @model_validator(mode="after")
    def validar_configuracion(self) -> "Configuracion":
        self.entorno = self.entorno.strip().lower()

        clave = self.secret_key.strip()
        if len(clave) < 32 or clave.lower().startswith(PREFIJOS_CLAVE_INSEGURA):
            raise ValueError("SECRET_KEY debe tener al menos 32 caracteres y no puede ser un valor de ejemplo")

        if self.algorithm not in ALGORITMOS_PERMITIDOS:
            raise ValueError(f"ALGORITHM debe ser uno de {sorted(ALGORITMOS_PERMITIDOS)}")

        if "*" in self.origenes_permitidos:
            raise ValueError("ORIGENES_PERMITIDOS no puede contener '*' cuando se usan cookies de sesión")

        if self.bcrypt_rounds < BCRYPT_RONDAS_MINIMAS:
            raise ValueError(f"BCRYPT_ROUNDS no puede ser menor a {BCRYPT_RONDAS_MINIMAS}")

        if self.db_pool_size < 1 or self.db_max_overflow < 0:
            raise ValueError("DB_POOL_SIZE debe ser >= 1 y DB_MAX_OVERFLOW >= 0")

        if not self.direct_url:
            raise ValueError("DIRECT_URL (usuario signafree_owner) es obligatorio")

        if self.entorno == "prod":
            self.debug = False
            if not self.cookie_secure:
                raise ValueError("COOKIE_SECURE debe ser true en producción")
            if not self.rls_activo:
                raise ValueError("RLS_ACTIVO debe ser true en producción")

        return self

    @property
    def url_app(self) -> str:
        return self.database_url

    @property
    def url_owner(self) -> str:
        return self.direct_url  # type: ignore[return-value]

    @property
    def google_cuenta_servicio_info(self) -> Optional[Dict[str, Any]]:
        crudo = (self.google_service_account_json_b64 or "").strip()
        if not crudo:
            return None
        return json.loads(base64.b64decode(crudo).decode("utf-8"))

    @property
    def allowed_origins(self) -> List[str]:
        return self.origenes_permitidos


configuracion = Configuracion()