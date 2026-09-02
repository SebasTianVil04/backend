from pydantic_settings import BaseSettings
from typing import List, Optional
from pydantic import validator

class Configuracion(BaseSettings):
    # Base de datos
    database_url: str
    
    # Archivos
    max_file_size: int = 10 * 1024 * 1024
    upload_dir: str = "archivos_subidos"
    temp_dir: str = "archivos_subidos/temp"
    
    # Google Drive
    usar_google_drive: bool = True
    google_drive_credenciales: str = "credenciales_drive.json"
    google_drive_carpeta_id: str = "1pKAI1bcy29M9Q915jDYmmmekj7r-Ok8B"
    
    # Seguridad
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # Token de recuperación
    reset_token_expire_minutes: int = 30
    
    # API Peru
    apiperu_token: str = "5782e7bdb7c9fc17b60a2285e26e08258c8b727c6181670349ba6825af50847f"
    apiperu_base_url: str = "https://apiperu.dev/api"
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # CORS
    allowed_origins: List[str] = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sebastianvil04.github.io",
        "http://38.56.216.65",
        "http://38.56.216.65:4200",
        "https://38.56.216.65"
    ]
    
    # Email/SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    mail_from: str = "SignaFree <noreply@signafree.com>"
    
    # Frontend
    frontend_url: str = "http://localhost:4200"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

configuracion = Configuracion()