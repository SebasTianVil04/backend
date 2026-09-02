# SignaFree Backend

API backend de **SignaFree**, una plataforma para el aprendizaje de la Lengua de Señas Peruana. El servicio expone funcionalidades de autenticación, lecciones, clases, prácticas, exámenes, seguimiento del progreso, gestión de datasets y reconocimiento de señas mediante modelos de inteligencia artificial.

La aplicación está construida con FastAPI y utiliza PostgreSQL como base de datos. También puede integrar Google Drive para almacenar archivos y servicios externos para la validación de datos y el envío de correo electrónico.

## Funcionalidades

- Registro, inicio de sesión, recuperación de contraseña y verificación de correo.
- Gestión de usuarios, perfiles, roles y permisos de administración.
- Organización del contenido mediante tipos de categoría, categorías, lecciones y clases.
- Prácticas, exámenes, resultados y estadísticas de aprendizaje.
- Registro de progreso, gamificación, ranking y tiempo de estudio.
- Carga, aprobación, rechazo y administración de videos para datasets.
- Entrenamiento, validación, activación y administración de modelos de inteligencia artificial.
- Traducción de señas a texto y reconocimiento de secuencias de video.
- Captura y procesamiento de videos.
- Almacenamiento local de archivos y almacenamiento opcional en Google Drive.

## Requisitos

- Python 3.8 o superior. Las versiones de NumPy, Pillow, MediaPipe y protobuf están fijadas para mantener compatibilidad con Python 3.8.
- PostgreSQL accesible desde la aplicación.
- FFmpeg disponible en el sistema para las operaciones de procesamiento de video.
- PyTorch instalado. `run.py` lo verifica al iniciar, pero actualmente no está incluido en `requirements.txt`; instálalo según tu sistema y plataforma (CPU o CUDA).
- Credenciales de Google Drive únicamente si se utilizará esa integración.

## Instalación

Desde la raíz del repositorio:

```bash
python -m venv .venv
```

Activar el entorno virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instalar PyTorch siguiendo el selector oficial de [pytorch.org](https://pytorch.org/get-started/locally/). Para una instalación únicamente con CPU, por ejemplo:

```bash
pip install torch torchvision
```

La instalación de MediaPipe, OpenCV y PyTorch puede depender de la versión de Python y del sistema operativo. Si alguna wheel no está disponible, utiliza una versión de Python compatible con las restricciones indicadas en `requirements.txt`.

## Configuración

La aplicación carga las variables desde un archivo `.env` ubicado en la raíz del proyecto. No existe un `.env.example` incluido actualmente; crea `.env` manualmente con, como mínimo:

```dotenv
DATABASE_URL=postgresql://usuario:password@localhost:5432/signafree
SECRET_KEY=construye-una-clave-larga-y-aleatoria
```

### Variables disponibles

| Variable | Obligatoria | Valor predeterminado | Descripción |
| --- | --- | --- | --- |
| `DATABASE_URL` | Sí | No tiene | URL de conexión a PostgreSQL. |
| `SECRET_KEY` | Sí | No tiene | Clave utilizada para firmar los tokens JWT. |
| `HOST` | No | `0.0.0.0` | Interfaz de red del servidor. |
| `PORT` | No | `8000` | Puerto del servidor. |
| `DEBUG` | No | `true` | Activa la recarga automática y detalles de error. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `1440` | Duración del token de acceso. |
| `RESET_TOKEN_EXPIRE_MINUTES` | No | `30` | Duración del token de recuperación. |
| `FRONTEND_URL` | No | `http://localhost:4200` | URL esperada del frontend. |
| `ALLOWED_ORIGINS` | No | Configurada en código | Orígenes permitidos para CORS. |
| `MAX_FILE_SIZE` | No | `10485760` | Tamaño máximo de archivo en bytes. |
| `UPLOAD_DIR` | No | `archivos_subidos` | Directorio de archivos subidos. |
| `TEMP_DIR` | No | `archivos_subidos/temp` | Directorio de archivos temporales. |
| `USAR_GOOGLE_DRIVE` | No | `true` | Activa el uso de Google Drive. |
| `GOOGLE_DRIVE_CREDENCIALES` | No | `credenciales_drive.json` | Archivo de credenciales de Google Drive. |
| `GOOGLE_DRIVE_CARPETA_ID` | No | Configurado en código | Carpeta raíz de Google Drive. |
| `APIPERU_TOKEN` | No | Configurado en código | Token del servicio API Perú. |
| `APIPERU_BASE_URL` | No | `https://apiperu.dev/api` | URL base de API Perú. |
| `SMTP_HOST` | No | `smtp.gmail.com` | Servidor SMTP. |
| `SMTP_PORT` | No | `587` | Puerto SMTP. |
| `SMTP_USER` | No | Vacío | Usuario de correo. |
| `SMTP_PASSWORD` | No | Vacío | Contraseña o clave de aplicación SMTP. |
| `MAIL_FROM` | No | `SignaFree <noreply@signafree.com>` | Remitente de los correos. |

Los nombres de las variables se leen sin distinguir mayúsculas y minúsculas mediante `pydantic-settings`. Para entornos distintos de desarrollo, define explícitamente todos los valores sensibles y desactiva `DEBUG`.

## Base de datos

El backend usa SQLAlchemy con PostgreSQL. Durante el evento de inicio, la aplicación importa los modelos y ejecuta `Base.metadata.create_all`, por lo que verifica o crea las tablas necesarias.

Para gestionar cambios de esquema con Alembic:

```bash
alembic upgrade head
```

Crear una nueva migración después de modificar los modelos:

```bash
alembic revision --autogenerate -m "descripcion del cambio"
alembic upgrade head
```

Comprueba que `DATABASE_URL` esté disponible en el entorno antes de ejecutar estos comandos.

## Ejecución

El script recomendado realiza verificaciones de entorno, crea los directorios necesarios, comprueba dependencias críticas e inicia Uvicorn:

```bash
python run.py
```

También puede iniciarse directamente con Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La aplicación crea o verifica, entre otros, estos directorios:

```text
archivos_subidos/
archivos_subidos/temp/
archivos_subidos/dataset_entrenamiento/
archivos_subidos/videos_lecciones/
archivos_subidos/imagenes_entrenamiento/
modelo_ia/datos_entrenamiento/
```

## Documentación y comprobación del servicio

Con el servidor ejecutándose:

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Health check: <http://localhost:8000/health>

Comprobación rápida desde una terminal:

```bash
curl http://localhost:8000/health
```

La respuesta esperada contiene el estado `saludable` y la versión `2.0.0`.

## Rutas principales

Todas las rutas de negocio se registran bajo el prefijo `/api/v1`.

| Módulo | Prefijo |
| --- | --- |
| Autenticación | `/api/v1/auth` |
| Usuarios | `/api/v1/usuarios` |
| Categorías y tipos de categoría | `/api/v1/categorias`, `/api/v1/tipos-categoria` |
| Lecciones y clases | `/api/v1/lecciones`, `/api/v1/clases` |
| Prácticas y progreso | `/api/v1/practicas`, `/api/v1/progreso` |
| Exámenes | `/api/v1/examenes` |
| Tiempo de estudio y estadísticas | `/api/v1/estudio`, `/api/v1/estadisticas` |
| Dataset de entrenamiento | `/api/v1/dataset`, `/api/v1/categorias-dataset` |
| Reconocimiento de video | `/api/v1/reconocimiento-video` |
| Captura de entrenamiento | `/api/v1/captura-entrenamiento` |
| Traductor | `/api/v1/traductor` |
| Administración | `/api/v1/admin` |
| Administración de exámenes | `/api/v1/admin/examenes` |

La lista completa de operaciones, parámetros y esquemas está disponible en Swagger UI y ReDoc.

## Archivos y modelos de IA

Los archivos locales se sirven mediante los montajes estáticos `/uploads` y `/archivos_subidos`. El proyecto contiene directorios para datasets, videos, imágenes, frames, archivos temporales y datos de entrenamiento.

Los modelos entrenados se almacenan en `modelos_entrenados/`. El código también permite cargar, validar, activar y combinar modelos para reconocimiento. Las tareas de entrenamiento pueden ser intensivas en CPU, memoria y almacenamiento, por lo que deben ejecutarse en un entorno preparado para procesamiento de video y aprendizaje automático.

Para Google Drive, coloca los archivos de autenticación en la raíz del proyecto con los nombres esperados por el código (`client_secret.json` y, después de autorizar, `token.json`). La primera autenticación puede abrir un flujo local del navegador.

## Estructura del proyecto

```text
app/
├── esquemas/       # Esquemas Pydantic de entrada y salida
├── modelos/        # Modelos SQLAlchemy y componentes de IA
├── rutas/          # Routers y endpoints de la API
├── servicios/      # Integraciones, entrenamiento y procesamiento
└── utilidades/     # Configuración, seguridad, validaciones y BD
alembic/            # Configuración y versiones de migraciones
archivos_subidos/   # Archivos recibidos y material de entrenamiento
modelo_ia/          # Datos de entrenamiento de IA
modelos_entrenados/ # Pesos de modelos entrenados
run.py              # Verificaciones e inicio del servidor
requirements.txt    # Dependencias Python fijadas
```

## Seguridad

- No subas a Git credenciales, tokens, contraseñas, `client_secret.json`, `credenciales_drive.json` ni `token.json`.
- Genera un `SECRET_KEY` único y suficientemente largo para cada entorno.
- Cambia cualquier credencial administrativa inicial antes de usar el sistema en un entorno real.
- Usa `DEBUG=false` en producción para no exponer trazas de errores en las respuestas.
- Revisa y restringe `ALLOWED_ORIGINS` antes de publicar el servicio.
- Protege los directorios estáticos si contienen archivos privados o datos de usuarios.
- Utiliza HTTPS y un gestor de secretos en despliegues públicos.

## Estado del proyecto

La API declara actualmente la versión `2.0.0`. El proyecto incluye una migración de Alembic y una colección de modelos entrenados, pero no incluye una suite de pruebas automatizadas en el repositorio. Se recomienda validar el arranque, la conexión a PostgreSQL y los flujos de autenticación antes de desplegar cambios.
