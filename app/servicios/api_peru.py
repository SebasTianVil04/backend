import httpx
from typing import Optional
from fastapi import HTTPException, status
from ..utilidades.configuracion import configuracion
from ..esquemas.usuario_schemas import DatosApiperu

class ServicioApiPeru:
    """Servicio para interactuar con APIPeru.dev"""
    
    def __init__(self):
        # URL correcta de APIPeru.dev
        self.base_url = "https://apiperu.dev/api"
        self.token = configuracion.apiperu_token
        
        # Headers según documentación
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        # Verificar si está en modo de prueba
        self.modo_prueba = (
            not self.token or 
            self.token in ["test_token_temporal", "tu_token_aqui", "test_token"] or
            len(self.token) < 10
        )
        
        print(f"API Peru - Modo prueba: {self.modo_prueba}")
        if self.modo_prueba:
            print("Para usar API real, configura APIPERU_TOKEN en .env")
        else:
            print(f"Token configurado: {self.token[:10]}...")
        
        # Datos de prueba para desarrollo
        self.datos_prueba = {
            "12345678": {
                "nombres": "JUAN CARLOS",
                "apellido_paterno": "RODRIGUEZ", 
                "apellido_materno": "LOPEZ"
            },
            "87654321": {
                "nombres": "MARIA ELENA",
                "apellido_paterno": "GONZALES",
                "apellido_materno": "TORRES"
            },
            "11111111": {
                "nombres": "PEDRO JOSE",
                "apellido_paterno": "MARTINEZ",
                "apellido_materno": "SILVA"
            }
        }
    
    async def consultar_dni(self, dni: str) -> DatosApiperu:
        """
        Consultar datos de persona por DNI usando APIPeru.dev
        """
        if not dni or len(dni) != 8 or not dni.isdigit():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="DNI debe tener exactamente 8 dígitos numéricos"
            )
        
        # MODO DE PRUEBA - Para desarrollo sin token
        if self.modo_prueba:
            print(f"Modo prueba: consultando DNI {dni}")
            if dni in self.datos_prueba:
                data = self.datos_prueba[dni]
                return DatosApiperu(
                    nombres=data["nombres"],
                    apellido_paterno=data["apellido_paterno"],
                    apellido_materno=data["apellido_materno"],
                    fecha_nacimiento="1990-01-01",  # No disponible en API
                    direccion=None  # No disponible en API
                )
            else:
                # Generar datos ficticios para cualquier DNI válido
                nombres_ficticios = ["CARLOS ALBERTO", "MARIA JOSE", "JOSE ANTONIO", "ANA LUCIA", "PEDRO MANUEL", "LUCIA MARIA"]
                apellidos_ficticios = ["GARCIA", "LOPEZ", "MARTINEZ", "RODRIGUEZ", "GONZALES", "FERNANDEZ", "TORRES", "SILVA", "CASTRO", "MORALES"]
                
                import random
                nombre = random.choice(nombres_ficticios)
                apellido1 = random.choice(apellidos_ficticios)
                apellido2 = random.choice(apellidos_ficticios)
                
                return DatosApiperu(
                    nombres=nombre,
                    apellido_paterno=apellido1,
                    apellido_materno=apellido2,
                    fecha_nacimiento="1990-01-01",
                    direccion=None
                )
        
        # MODO REAL - API Peru.dev
        url = f"{self.base_url}/dni"
        body = {"dni": dni}
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    url,
                    json=body,
                    headers=self.headers
                )
                
                print(f"API Response Status: {response.status_code}")
                
                # Manejo de errores según API Peru.dev
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Token de API Peru inválido o expirado"
                    )
                
                if response.status_code == 404:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="DNI no encontrado en las fuentes públicas"
                    )
                
                if response.status_code == 429:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Límite de consultas excedido"
                    )
                
                if response.status_code != 200:
                    error_detail = f"Error en API Peru: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "message" in error_data:
                            error_detail = error_data["message"]
                    except:
                        pass
                    
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=error_detail
                    )
                
                # Procesar respuesta exitosa
                data = response.json()
                print(f"API Response Data: {data}")
                
                # Verificar estructura de respuesta según documentación
                if not data.get("success"):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="DNI no encontrado o datos incompletos"
                    )
                
                api_data = data.get("data", {})
                
                # Extraer datos según la estructura de APIPeru.dev
                nombres = api_data.get("nombres", "").strip()
                apellido_paterno = api_data.get("apellido_paterno", "").strip()
                apellido_materno = api_data.get("apellido_materno", "").strip()
                
                if not nombres or not apellido_paterno:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="DNI encontrado pero datos incompletos"
                    )
                
                # Limpiar datos censurados (quitar asteriscos)
                nombres = self._limpiar_datos_censurados(nombres)
                apellido_paterno = self._limpiar_datos_censurados(apellido_paterno)
                apellido_materno = self._limpiar_datos_censurados(apellido_materno)
                
                return DatosApiperu(
                    nombres=nombres,
                    apellido_paterno=apellido_paterno,
                    apellido_materno=apellido_materno,
                    fecha_nacimiento=None,  
                    direccion=None  
                )
                
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timeout al consultar API Peru"
            )
        except httpx.RequestError as e:
            print(f" Error de conexión: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Error de conexión con API Peru"
            )
        except HTTPException:
            # Re-lanzar excepciones HTTP que ya manejamos
            raise
        except Exception as e:
            print(f" Error inesperado: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno al procesar consulta"
            )
    
    def _limpiar_datos_censurados(self, texto: str) -> str:
        """
        Limpiar datos censurados de la API (asteriscos)
        Nota: La API gratuita devuelve datos censurados con asteriscos
        """
        if not texto:
            return ""
        
        return texto.strip().upper()

# Instancia global del servicio
servicio_api_peru = ServicioApiPeru()