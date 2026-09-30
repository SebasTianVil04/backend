from google_auth_oauthlib.flow import InstalledAppFlow

from app.utilidades.configuracion import configuracion

SCOPES = ["https://www.googleapis.com/auth/drive"]

flujo = InstalledAppFlow.from_client_config(
    {
        "installed": {
            "client_id": configuracion.google_client_id,
            "client_secret": configuracion.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": configuracion.google_token_uri,
            "redirect_uris": ["http://localhost"],
        }
    },
    SCOPES,
)

credenciales = flujo.run_local_server(port=0, access_type="offline", prompt="consent")
print(credenciales.refresh_token)