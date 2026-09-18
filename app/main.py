from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import threading
from pathlib import Path

from app.utilidades.configuracion import configuracion
from app.utilidades.base_datos import crear_tablas


def log(mensaje: str):
    print(mensaje, flush=True)


routers_disponibles = {}

log("Importando routers...")

try:
    from app.rutas import autenticacion
    log("  autenticacion OK")
    routers_disponibles["autenticacion"] = autenticacion
except Exception as e:
    log(f"  Error autenticacion: {e}")

try:
    from app.rutas import rutas_captura
    log("  rutas_captura OK")
    routers_disponibles["rutas_captura"] = rutas_captura
except Exception as e:
    log(f"  Error rutas_captura: {e}")

try:
    from app.rutas import usuarios
    log("  usuarios OK")
    routers_disponibles["usuarios"] = usuarios
except Exception as e:
    log(f"  Error usuarios: {e}")

try:
    from app.rutas import categorias
    log("  categorias OK")
    routers_disponibles["categorias"] = categorias
except Exception as e:
    log(f"  Error categorias: {e}")

try:
    from app.rutas import lecciones
    log("  lecciones OK")
    routers_disponibles["lecciones"] = lecciones
except Exception as e:
    log(f"  Error lecciones: {e}")

try:
    from app.rutas import clases
    log("  clases OK")
    routers_disponibles["clases"] = clases
except Exception as e:
    log(f"  Error clases: {e}")

try:
    from app.rutas import practicas
    log("  practicas OK")
    routers_disponibles["practicas"] = practicas
except Exception as e:
    log(f"  Error practicas: {e}")

try:
    from app.rutas import progreso
    log("  progreso OK")
    routers_disponibles["progreso"] = progreso
except Exception as e:
    log(f"  Error progreso: {e}")

try:
    from app.rutas import examenes
    log("  examenes OK")
    routers_disponibles["examenes"] = examenes
except Exception as e:
    log(f"  Error examenes: {e}")

try:
    from app.rutas import dataset
    log("  dataset OK")
    routers_disponibles["dataset"] = dataset
except Exception as e:
    log(f"  Error dataset: {e}")

try:
    from app.rutas import traductor
    log("  traductor OK")
    routers_disponibles["traductor"] = traductor
except Exception as e:
    log(f"  Error traductor: {e}")

try:
    from app.rutas import admin
    log("  admin OK")
    routers_disponibles["admin"] = admin
except Exception as e:
    log(f"  Error admin: {e}")

try:
    from app.rutas import examenes_admin
    log("  examenes_admin OK")
    routers_disponibles["examenes_admin"] = examenes_admin
except Exception as e:
    log(f"  Error examenes_admin: {e}")

try:
    from app.rutas import estudio
    log("  estudio OK")
    routers_disponibles["estudio"] = estudio
except Exception as e:
    log(f"  Error estudio: {e}")

try:
    from app.rutas import estadisticas_rutas
    log("  estadisticas_rutas OK")
    routers_disponibles["estadisticas_rutas"] = estadisticas_rutas
except Exception as e:
    log(f"  Error estadisticas_rutas: {e}")

try:
    from app.rutas import reconocimiento_video
    log("  reconocimiento_video OK")
    routers_disponibles["reconocimiento_video"] = reconocimiento_video
except Exception as e:
    log(f"  Error reconocimiento_video: {e}")

try:
    from app.rutas import tipos_categoria
    log("  tipos_categoria OK")
    routers_disponibles["tipos_categoria"] = tipos_categoria
except Exception as e:
    log(f"  Error tipos_categoria: {e}")

try:
    from app.rutas import categorias_dataset
    log("  categorias_dataset OK")
    routers_disponibles["categorias_dataset"] = categorias_dataset
except Exception as e:
    log(f"  Error categorias_dataset: {e}")

try:
    from app.rutas import senas_categoria
    log("  senas_categoria OK")
    routers_disponibles["senas_categoria"] = senas_categoria
except Exception as e:
    log(f"  Error senas_categoria: {e}")

try:
    from app.rutas import roles
    log("  roles OK")
    routers_disponibles["roles"] = roles
except Exception as e:
    log(f"  Error roles: {e}")

try:
    from app.rutas import menu
    log("  menu OK")
    routers_disponibles["menu"] = menu
except Exception as e:
    log(f"  Error menu: {e}")

try:
    from app.rutas import menu_admin
    log("  menu_admin OK")
    routers_disponibles["menu_admin"] = menu_admin
except Exception as e:
    log(f"  Error menu_admin: {e}")


log("Todos los routers importados")

app = FastAPI(
    title="SignaFree API",
    description="API para aprendizaje de Lengua de Señas Peruana",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False
)

log("Configurando CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=configuracion.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"],
    max_age=3600,
)
log("CORS configurado")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    method = request.method
    path = request.url.path

    log(f"IN {method} {path} | Origin: {origin}")

    response = await call_next(request)

    log(f"OUT {method} {path} | Status: {response.status_code}")

    return response


def crear_directorios():
    directorios = [
        configuracion.upload_dir,
        configuracion.temp_dir,
        f"{configuracion.upload_dir}/dataset_entrenamiento",
        f"{configuracion.upload_dir}/videos_lecciones",
        f"{configuracion.upload_dir}/imagenes_entrenamiento",
        "archivos_subidos",
        "archivos_subidos/videos_dataset",
        "archivos_subidos/frames_dataset",
        "archivos_subidos/senas_referencia"
    ]

    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        log(f"Directorio creado/verificado: {directorio}")


def crear_tablas_con_timeout(timeout_segundos: int = 15):
    resultado = {"error": None, "completado": False}

    def objetivo():
        try:
            crear_tablas()
            resultado["completado"] = True
        except Exception as e:
            resultado["error"] = e

    hilo = threading.Thread(target=objetivo, daemon=True)
    hilo.start()
    hilo.join(timeout=timeout_segundos)

    if hilo.is_alive():
        log(f"ADVERTENCIA: crear_tablas() no respondió en {timeout_segundos}s.")
        log("Probable causa: la base de datos no está accesible (host/puerto/credenciales) o la conexión está colgada.")
        return False

    if resultado["error"] is not None:
        log(f"Error en base de datos: {resultado['error']}")
        return False

    return resultado["completado"]


import logging
logging.basicConfig(level=logging.INFO)


def sembrar_datos_iniciales():
    """
    Orden OBLIGATORIO: permisos -> roles -> admin -> menú.
    - seed_permisos crea el catálogo de permisos en BD.
    - seed_roles crea los roles y resuelve sus permisos por código
      (necesita que los permisos ya existan).
    - seed_admin crea/asegura el administrador principal, leyendo
      credenciales de variables de entorno (ADMIN_EMAIL, ADMIN_PASSWORD,
      etc.). Necesita que el rol "admin" ya exista. Si faltan las
      variables de entorno, se omite con un warning sin romper el arranque.
    - seed_menu crea los ítems de menú (referencian permisos por código).
    """
    try:
        from app.utilidades.base_datos import SessionLocal
    except Exception as e:
        log(f"No se pudo importar SessionLocal: {e}")
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

    db = SessionLocal()
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


@app.on_event("startup")
async def startup_event():
    log("=" * 50)
    log("INICIANDO SIGNAFREE API v2.0.0")
    log("=" * 50)

    try:
        crear_directorios()
        log("Directorios verificados")
    except Exception as e:
        log(f"Error creando directorios: {e}")

    if crear_tablas_con_timeout():
        log("Base de datos verificada")
        sembrar_datos_iniciales()
    else:
        log("Continuando sin confirmar la base de datos (revisa la conexión).")

    log("SIGNAFREE API LISTA")
    log(f"Servidor: http://{configuracion.host}:{configuracion.port}")
    log(f"Documentacion: http://{configuracion.host}:{configuracion.port}/docs")
    log("=" * 50)


try:
    upload_path = Path(configuracion.upload_dir)
    if upload_path.exists():
        app.mount("/uploads", StaticFiles(directory=str(upload_path)), name="uploads")
        log(f"Archivos estaticos montados en /uploads desde: {upload_path}")
    else:
        log(f"Directorio de uploads no existe: {upload_path}")

    archivos_subidos_path = Path("archivos_subidos")
    if not archivos_subidos_path.exists():
        archivos_subidos_path.mkdir(exist_ok=True)
        log(f"Directorio archivos_subidos creado: {archivos_subidos_path}")

    app.mount("/archivos_subidos", StaticFiles(directory=str(archivos_subidos_path)), name="archivos_subidos")
    log(f"Archivos estaticos montados en /archivos_subidos desde: {archivos_subidos_path}")

    senas_referencia_path = archivos_subidos_path / "senas_referencia"
    senas_referencia_path.mkdir(parents=True, exist_ok=True)
    app.mount("/archivos/senas_referencia", StaticFiles(directory=str(senas_referencia_path)), name="senas_referencia")
    log(f"Archivos estaticos montados en /archivos/senas_referencia desde: {senas_referencia_path}")

except Exception as e:
    log(f"Error montando archivos estaticos: {e}")
    import traceback
    traceback.print_exc()

log("Registrando rutas...")

if "autenticacion" in routers_disponibles:
    app.include_router(routers_disponibles["autenticacion"].router, prefix="/api/v1")
    log("  Autenticacion")

if "usuarios" in routers_disponibles:
    app.include_router(routers_disponibles["usuarios"].router, prefix="/api/v1")
    log("  Usuarios")

if "categorias" in routers_disponibles:
    app.include_router(routers_disponibles["categorias"].router, prefix="/api/v1")
    log("  Categorias")

if "lecciones" in routers_disponibles:
    app.include_router(routers_disponibles["lecciones"].router, prefix="/api/v1")
    log("  Lecciones")

if "clases" in routers_disponibles:
    app.include_router(routers_disponibles["clases"].router, prefix="/api/v1")
    log("  Clases")

if "practicas" in routers_disponibles:
    app.include_router(routers_disponibles["practicas"].router, prefix="/api/v1")
    log("  Practicas")

if "progreso" in routers_disponibles:
    app.include_router(routers_disponibles["progreso"].router, prefix="/api/v1")
    log("  Progreso")

if "examenes" in routers_disponibles:
    app.include_router(routers_disponibles["examenes"].router, prefix="/api/v1")
    log("  Examenes")

if "dataset" in routers_disponibles:
    app.include_router(routers_disponibles["dataset"].router, prefix="/api/v1")
    log("  Dataset")

if "reconocimiento_video" in routers_disponibles:
    app.include_router(routers_disponibles["reconocimiento_video"].router, prefix="/api/v1")
    log("  Reconocimiento")

if "traductor" in routers_disponibles:
    app.include_router(routers_disponibles["traductor"].router, prefix="/api/v1")
    log("  Traductor")

if "admin" in routers_disponibles:
    app.include_router(routers_disponibles["admin"].router, prefix="/api/v1")
    log("  Admin")

if "rutas_captura" in routers_disponibles:
    app.include_router(routers_disponibles["rutas_captura"].router, prefix="/api/v1")
    log("  Captura")

if "examenes_admin" in routers_disponibles:
    app.include_router(routers_disponibles["examenes_admin"].router, prefix="/api/v1/admin/examenes")
    log("  Examenes Admin")

if "estudio" in routers_disponibles:
    app.include_router(routers_disponibles["estudio"].router, prefix="/api/v1")
    log("  Estudio")

if "estadisticas_rutas" in routers_disponibles:
    app.include_router(routers_disponibles["estadisticas_rutas"].router, prefix="/api/v1")
    log("  Estadisticas")

if "tipos_categoria" in routers_disponibles:
    app.include_router(routers_disponibles["tipos_categoria"].router, prefix="/api/v1")
    log("  Tipos Categoria")

if "categorias_dataset" in routers_disponibles:
    app.include_router(routers_disponibles["categorias_dataset"].router, prefix="/api/v1")
    log("  Categorias Dataset")

if "senas_categoria" in routers_disponibles:
    app.include_router(routers_disponibles["senas_categoria"].router, prefix="/api/v1")
    log("  Senas Categoria")

if "roles" in routers_disponibles:
    app.include_router(routers_disponibles["roles"].router)
    log("  Roles")
else:
    log("  Roles (omitido por error de importación)")

if "menu" in routers_disponibles:
    app.include_router(routers_disponibles["menu"].router, prefix="/api/v1")
    log("  Menu (usuario)")
else:
    log("  Menu (omitido por error de importación)")

if "menu_admin" in routers_disponibles:
    app.include_router(routers_disponibles["menu_admin"].router)
    log("  Menu Admin")
else:
    log("  Menu Admin (omitido por error de importación)")

log("Todas las rutas registradas")


@app.get("/", tags=["General"])
async def root():
    return {
        "nombre": "SignaFree API",
        "version": "2.0.0",
        "estado": "activo",
        "mensaje": "Bienvenido a SignaFree API",
        "documentacion": "/docs",
        "salud": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    return {
        "estado": "saludable",
        "version": "2.0.0",
        "servicios": {
            "api": "funcionando",
            "base_datos": "conectado",
            "cors": "habilitado"
        }
    }


@app.get("/api/v1/test-cors", tags=["General"])
async def test_cors():
    return {
        "mensaje": "CORS funcionando correctamente",
        "timestamp": "2025-01-01T00:00:00Z"
    }


@app.get("/api/v1/test-archivos", tags=["General"])
async def test_archivos():
    archivos_subidos_path = Path("archivos_subidos")
    archivos = []

    if archivos_subidos_path.exists():
        for root, dirs, files in os.walk(archivos_subidos_path):
            for file in files:
                relative_path = Path(root) / file
                archivos.append(str(relative_path.relative_to(archivos_subidos_path)))

    return {
        "directorio_archivos_subidos": str(archivos_subidos_path.absolute()),
        "archivos_encontrados": archivos,
        "url_ejemplo": "http://localhost:8000/archivos_subidos/videos_dataset/A_20251031_211516_683467.webm"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "mensaje": exc.detail,
            "codigo": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    import traceback
    error_traceback = traceback.format_exc()

    log(f"Error no manejado: {str(exc)}")
    log(error_traceback)

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "mensaje": f"Error interno: {str(exc)}",
            "detalle": error_traceback if configuracion.debug else None,
            "codigo": 500
        }
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
        reload=configuracion.debug,
        log_level="info"
    )