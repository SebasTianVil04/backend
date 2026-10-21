import os
import importlib
import threading
import traceback

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.utilidades.configuracion import configuracion
from app.utilidades.base_datos import crear_tablas
from app.utilidades.seguridad import verificar_csrf


def log(mensaje: str):
    print(mensaje, flush=True)


ES_PRODUCCION = configuracion.entorno == "prod"
DEBUG_ACTIVO = bool(configuracion.debug) and not ES_PRODUCCION

PREFIJO_ARCHIVO_FIRMADO = "/api/v1/archivos/f/"
PREFIJOS_SIN_CACHE = (
    "/api/v1/auth",
    "/api/v1/archivos/firmar",
    "/api/v1/archivos/video-dataset",
    "/api/v1/exportacion-dataset",
)

REGISTRO_ROUTERS = [
    ("autenticacion", "/api/v1"),
    ("usuarios", "/api/v1"),
    ("categorias", "/api/v1"),
    ("lecciones", "/api/v1"),
    ("clases", "/api/v1"),
    ("practicas", "/api/v1"),
    ("progreso", "/api/v1"),
    ("examenes", "/api/v1"),
    ("dataset", "/api/v1"),
    ("reconocimiento_video", "/api/v1"),
    ("traductor", "/api/v1"),
    ("admin", "/api/v1"),
    ("rutas_captura", "/api/v1"),
    ("examenes_admin", "/api/v1/admin/examenes"),
    ("estudio", "/api/v1"),
    ("estadisticas_rutas", "/api/v1"),
    ("tipos_categoria", "/api/v1"),
    ("categorias_dataset", "/api/v1"),
    ("exportacion_dataset", "/api/v1"),
    ("senas_categoria", "/api/v1"),
    ("roles", ""),
    ("menu", "/api/v1"),
    ("menu_admin", ""),
    ("contacto", "/api/v1"),
    ("tickets", "/api/v1"),
    ("tickets_admin", "/api/v1"),
    ("archivos", "/api/v1"),
]

routers_disponibles = {}

log("Importando routers...")
for nombre, _ in REGISTRO_ROUTERS:
    try:
        routers_disponibles[nombre] = importlib.import_module(f"app.rutas.{nombre}")
        log(f"  {nombre} OK")
    except Exception as e:
        log(f"  Error {nombre}: {e}")
log("Todos los routers importados")

app = FastAPI(
    title="SignaFree API",
    description="API para aprendizaje de Lengua de Señas Peruana",
    version="2.0.0",
    docs_url=None if ES_PRODUCCION else "/docs",
    redoc_url=None if ES_PRODUCCION else "/redoc",
    openapi_url=None if ES_PRODUCCION else "/openapi.json",
    redirect_slashes=False,
    dependencies=[Depends(verificar_csrf)],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    method = request.method
    path = request.url.path

    if path.startswith(PREFIJO_ARCHIVO_FIRMADO):
        path = f"{PREFIJO_ARCHIVO_FIRMADO}[token]"

    log(f"IN {method} {path} | Origin: {origin}")
    response = await call_next(request)
    log(f"OUT {method} {path} | Status: {response.status_code}")
    return response


@app.middleware("http")
async def cabeceras_seguridad(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if ES_PRODUCCION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.url.path.startswith(PREFIJOS_SIN_CACHE):
        response.headers["Cache-Control"] = "no-store"
    return response


log("Configurando CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=configuracion.origenes_permitidos,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-CSRF-Token",
        "Accept",
        "Accept-Language",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=["Content-Length", "Content-Disposition"],
    max_age=600,
)
log("CORS configurado")


def crear_directorios():
    directorios = [
        configuracion.upload_dir,
        configuracion.temp_dir,
        f"{configuracion.upload_dir}/dataset_entrenamiento",
        f"{configuracion.upload_dir}/videos_lecciones",
        f"{configuracion.upload_dir}/imagenes_entrenamiento",
        f"{configuracion.upload_dir}/avatares_usuarios",
        "archivos_subidos",
        "archivos_subidos/videos_dataset",
        "archivos_subidos/frames_dataset",
        "archivos_subidos/senas_referencia",
        "archivos_subidos/secuencias_dataset",
    ]

    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        log(f"Directorio creado/verificado: {directorio}")


def sembrar_datos_iniciales():
    try:
        from app.utilidades.base_datos import SessionOwner
    except Exception as e:
        log(f"No se pudo importar SessionOwner: {e}")
        return

    try:
        from app.semillas.seed_permisos import seed_permisos
    except Exception as e:
        log(f"Seed permisos no disponible: {e}")
        seed_permisos = None

    try:
        from app.semillas.seed_roles import seed_roles
    except Exception as e:
        log(f"Seed roles no disponible: {e}")
        seed_roles = None

    try:
        from app.semillas.seed_admin import seed_admin
    except Exception as e:
        log(f"Seed admin no disponible: {e}")
        seed_admin = None

    try:
        from app.semillas.seed_menu import seed_menu
    except Exception as e:
        log(f"Seed menú no disponible: {e}")
        seed_menu = None

    db = SessionOwner()
    try:
        if seed_permisos:
            seed_permisos(db)
            log("Permisos sembrados")
        if seed_roles:
            seed_roles(db)
            log("Roles sembrados")
        if seed_admin:
            seed_admin(db)
            log("Admin principal verificado/sembrado")
        if seed_menu:
            seed_menu(db)
            log("Menú sembrado")
    except Exception as e:
        log(f"Error sembrando datos iniciales: {e}")
        db.rollback()
    finally:
        db.close()


def inicializar_base_de_datos(timeout_segundos: int = 60):
    resultado = {"error": None, "completado": False}

    def objetivo():
        try:
            crear_tablas()
            sembrar_datos_iniciales()
            resultado["completado"] = True
        except Exception as e:
            resultado["error"] = e

    hilo = threading.Thread(target=objetivo, daemon=True)
    hilo.start()
    hilo.join(timeout=timeout_segundos)

    if hilo.is_alive():
        log(f"ADVERTENCIA: la inicialización de la BD no terminó en {timeout_segundos}s.")
        log("El hilo sigue corriendo en segundo plano; los seeds se ejecutarán cuando termine.")
        return False

    if resultado["error"] is not None:
        log(f"Error en base de datos: {resultado['error']}")
        return False

    return resultado["completado"]


@app.on_event("startup")
async def startup_event():
    log("=" * 50)
    log("INICIANDO SIGNAFREE API v2.0.0")
    log("=" * 50)

    if ES_PRODUCCION and configuracion.debug:
        log("ADVERTENCIA: DEBUG está activo en producción; se ignora y se fuerza desactivado.")

    try:
        crear_directorios()
        log("Directorios verificados")
    except Exception as e:
        log(f"Error creando directorios: {e}")

    if inicializar_base_de_datos(timeout_segundos=60):
        log("Base de datos verificada y datos iniciales sembrados")
    else:
        log("La inicialización sigue en curso o falló. Revisa los logs.")

    log("SIGNAFREE API LISTA")
    log(f"Servidor: http://{configuracion.host}:{configuracion.port}")
    if not ES_PRODUCCION:
        log(f"Documentacion: http://{configuracion.host}:{configuracion.port}/docs")
    log("=" * 50)


log("Registrando rutas...")
for nombre, prefijo in REGISTRO_ROUTERS:
    modulo = routers_disponibles.get(nombre)
    if modulo is None:
        log(f"  {nombre} (omitido por error de importación)")
        continue
    if prefijo:
        app.include_router(modulo.router, prefix=prefijo)
    else:
        app.include_router(modulo.router)
    log(f"  {nombre}")
log("Todas las rutas registradas")


@app.get("/", tags=["General"])
async def root():
    respuesta = {
        "nombre": "SignaFree API",
        "version": "2.0.0",
        "estado": "activo",
        "mensaje": "Bienvenido a SignaFree API",
        "salud": "/health",
    }
    if not ES_PRODUCCION:
        respuesta["documentacion"] = "/docs"
    return respuesta


@app.get("/health", tags=["General"])
async def health_check():
    return {"estado": "saludable", "version": "2.0.0"}


@app.get("/api/v1/test-cors", tags=["General"])
async def test_cors():
    return {"mensaje": "CORS funcionando correctamente"}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "mensaje": exc.detail,
            "codigo": exc.status_code,
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_traceback = traceback.format_exc()

    log(f"Error no manejado: {str(exc)}")
    log(error_traceback)

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "mensaje": f"Error interno: {str(exc)}" if DEBUG_ACTIVO else "Error interno del servidor",
            "detalle": error_traceback if DEBUG_ACTIVO else None,
            "codigo": 500,
        },
    )


if __name__ == "__main__":
    import uvicorn

    log("=" * 50)
    log("Iniciando servidor con Uvicorn...")
    log("=" * 50)

    uvicorn.run(
        "app.main:app",
        host=configuracion.host,
        port=configuracion.port,
        reload=DEBUG_ACTIVO,
        log_level="info",
    )