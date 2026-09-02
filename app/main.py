from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import sys

from app.utilidades.configuracion import configuracion
from app.utilidades.base_datos import crear_tablas

print("Importando routers...")
try:
    from app.rutas import autenticacion
    print("  autenticacion OK")
except Exception as e:
    print(f"  Error autenticacion: {e}")
    sys.exit(1)

try:
    from app.rutas import rutas_captura
    print("  rutas_captura OK")
except Exception as e:
    print(f"  Error rutas_captura: {e}")

try:
    from app.rutas import usuarios
    print("  usuarios OK")
except Exception as e:
    print(f"  Error usuarios: {e}")

try:
    from app.rutas import categorias
    print("  categorias OK")
except Exception as e:
    print(f"  Error categorias: {e}")

try:
    from app.rutas import lecciones
    print("  lecciones OK")
except Exception as e:
    print(f"  Error lecciones: {e}")

try:
    from app.rutas import clases
    print("  clases OK")
except Exception as e:
    print(f"  Error clases: {e}")

try:
    from app.rutas import practicas
    print("  practicas OK")
except Exception as e:
    print(f"  Error practicas: {e}")

try:
    from app.rutas import progreso
    print("  progreso OK")
except Exception as e:
    print(f"  Error progreso: {e}")

try:
    from app.rutas import examenes
    print("  examenes OK")
except Exception as e:
    print(f"  Error examenes: {e}")

try:
    from app.rutas import dataset
    print("  dataset OK")
except Exception as e:
    print(f"  Error dataset: {e}")

try:
    from app.rutas import traductor
    print("  traductor OK")
except Exception as e:
    print(f"  Error traductor: {e}")

try:
    from app.rutas import admin
    print("  admin OK")
except Exception as e:
    print(f"  Error admin: {e}")

try:
    from app.rutas import examenes_admin
    print("  examenes_admin OK")
except Exception as e:
    print(f"  Error examenes_admin: {e}")

try:
    from app.rutas import estudio
    print("  estudio OK")
except Exception as e:
    print(f"  Error estudio: {e}")

try:
    from app.rutas import estadisticas_rutas
    print("  estadisticas_rutas OK")
except Exception as e:
    print(f"  Error estadisticas_rutas: {e}")

try:
    from app.rutas import reconocimiento_video
    print("  reconocimiento_video OK")
except Exception as e:
    print(f"  Error reconocimiento_video: {e}")

try:
    from app.rutas import tipos_categoria
    print("  tipos_categoria OK")
except Exception as e:
    print(f"  Error tipos_categoria: {e}")

try:
    from app.rutas import categorias_dataset
    print("  categorias_dataset OK")
except Exception as e:
    print(f"  Error categorias_dataset: {e}")


print("Todos los routers importados")

app = FastAPI(
    title="SignaFree API",
    description="API para aprendizaje de Lengua de Señas Peruana",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

print("Configurando CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://sebastianvil04.github.io",
        "http://38.56.216.65",
        "http://38.56.216.65:4200",
        "https://38.56.216.65"

    ],
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
print("CORS configurado")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    method = request.method
    path = request.url.path
    
    print(f"IN {method} {path} | Origin: {origin}")
    
    response = await call_next(request)
    
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With"
    
    print(f"OUT {method} {path} | Status: {response.status_code}")
    
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
        "archivos_subidos/frames_dataset"
    ]
    
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        print(f"Directorio creado/verificado: {directorio}")

def crear_usuario_admin_por_defecto():
    from .utilidades.base_datos import SessionLocal
    from .modelos.usuario import Usuario
    from .utilidades.seguridad import obtener_hash_password
    from datetime import date
    
    db = SessionLocal()
    
    try:
        usuario_existente = db.query(Usuario).filter(
            Usuario.email == "svilchezviera1704@gmail.com"
        ).first()
        
        if usuario_existente:
            print(f"Usuario ya existe: {usuario_existente.email}")
            print(f"Es admin: {usuario_existente.es_admin}")
            return
        
        print("\n" + "=" * 60)
        print("CREANDO USUARIO ADMINISTRADOR POR DEFECTO")
        print("=" * 60)
        
        admin = Usuario(
            tipo_usuario="peruano_mayor",
            email="svilchezviera1704@gmail.com",
            password_hash=obtener_hash_password("17Alexander%"),
            dni="76009799",
            nombres="SEBASTIAN ALEXANDER",
            apellido_paterno="VILCHEZ",
            apellido_materno="VIERA",
            telefono="+51940964458",
            direccion="Ah 12 de Octubre 123, Lima, Peru",
            fecha_nacimiento=date(2001, 4, 17),
            activo=True,
            es_admin=True,
            verificado=True
        )
        
        db.add(admin)
        db.commit()
        db.refresh(admin)
        
        print("\nUSUARIO ADMINISTRADOR CREADO EXITOSAMENTE")
        print("=" * 60)
        print(f"Email: {admin.email}")
        print(f"Password: 17Alexander%")
        print(f"ID: {admin.id}")
        print(f"Nombres: {admin.nombre_completo}")
        print(f"Es admin: {admin.es_admin}")
        print("=" * 60)
        print("IMPORTANTE: Cambia esta contraseña despues del primer inicio de sesion")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nERROR al crear usuario administrador: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("INICIANDO SIGNAFREE API v2.0.0")
    print("=" * 50)
    
    try:
        crear_directorios()
        print("Directorios verificados")
    except Exception as e:
        print(f"Error creando directorios: {e}")
    
    try:
        crear_tablas()
        print("Base de datos verificada")
    except Exception as e:
        print(f"Error en base de datos: {e}")
    
    try:
        crear_usuario_admin_por_defecto()
    except Exception as e:
        print(f"Error al crear usuario administrador: {e}")
    
    print("SIGNAFREE API LISTA")
    print(f"Servidor: http://{configuracion.host}:{configuracion.port}")
    print(f"Documentacion: http://{configuracion.host}:{configuracion.port}/docs")
    print("=" * 50)

try:
    upload_path = Path(configuracion.upload_dir)
    if upload_path.exists():
        app.mount("/uploads", StaticFiles(directory=str(upload_path)), name="uploads")
        print(f"Archivos estaticos montados en /uploads desde: {upload_path}")
    else:
        print(f"Directorio de uploads no existe: {upload_path}")
    
    archivos_subidos_path = Path("archivos_subidos")
    if archivos_subidos_path.exists():
        app.mount("/archivos_subidos", StaticFiles(directory=str(archivos_subidos_path)), name="archivos_subidos")
        print(f"Archivos estaticos montados en /archivos_subidos desde: {archivos_subidos_path}")
    else:
        print(f"Directorio archivos_subidos no existe: {archivos_subidos_path}")
        archivos_subidos_path.mkdir(exist_ok=True)
        print(f"Directorio archivos_subidos creado: {archivos_subidos_path}")
        
except Exception as e:
    print(f"Error montando archivos estaticos: {e}")
    import traceback
    traceback.print_exc()

@app.options("/api/v1/progreso/clases/{clase_id}/practica")
async def options_practica(clase_id: int):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
    )

print("Registrando rutas...")
app.include_router(autenticacion.router, prefix="/api/v1")
print("  Autenticacion")

app.include_router(usuarios.router, prefix="/api/v1")
print("  Usuarios")

app.include_router(categorias.router, prefix="/api/v1")
print("  Categorias")

app.include_router(lecciones.router, prefix="/api/v1")
print("  Lecciones")

app.include_router(clases.router, prefix="/api/v1")
print("  Clases")

app.include_router(practicas.router, prefix="/api/v1")
print("  Practicas")

app.include_router(progreso.router, prefix="/api/v1")
print("  Progreso")

app.include_router(examenes.router, prefix="/api/v1")
print("  Examenes")

app.include_router(dataset.router, prefix="/api/v1")
print("  Dataset")

app.include_router(reconocimiento_video.router, prefix="/api/v1")
print("  Reconocimiento")

app.include_router(traductor.router, prefix="/api/v1")
print("  Traductor")

app.include_router(admin.router, prefix="/api/v1")
print("  Admin")

app.include_router(rutas_captura.router, prefix="/api/v1")
print("  Captura")

app.include_router(examenes_admin.router, prefix="/api/v1/admin/examenes")
print("  Examenes Admin")

app.include_router(estudio.router, prefix="/api/v1")
print("  Estudio")

app.include_router(estadisticas_rutas.router, prefix="/api/v1")
print("  Estadisticas")

app.include_router(tipos_categoria.router, prefix="/api/v1")
print("  Tipos Categoria")

app.include_router(categorias_dataset.router, prefix="/api/v1")
print("  Categorias Dataset")

print("Todas las rutas registradas")

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
        },
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    import traceback
    error_traceback = traceback.format_exc()
    
    print(f"Error no manejado: {str(exc)}")
    print(error_traceback)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "mensaje": f"Error interno: {str(exc)}",
            "detalle": error_traceback if configuracion.debug else None,
            "codigo": 500
        },
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true",
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("Iniciando servidor con Uvicorn...")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=configuracion.host,
        port=configuracion.port,
        reload=configuracion.debug,
        log_level="info"
    )