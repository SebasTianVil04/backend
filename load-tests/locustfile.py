# load-tests/locustfile.py
import json
from locust import HttpUser, task, between, SequentialTaskSet

class SignaFreeUser(SequentialTaskSet):
    """
    Simula el flujo de un usuario en SignaFree:
    login -> ver categorías -> ver lecciones -> ver clases.
    """
    def on_start(self):
        """Se ejecuta una vez al inicio de cada usuario virtual."""
        # Reemplaza con un usuario de prueba real de tu base de datos
        self.email = "test.qa@correo.com"
        self.password = "Segura#2026"

        # 1. Iniciar sesión para obtener el token (si usas cookies, se manejan solas)
        with self.client.post("/api/v1/auth/login",
                              json={"email": self.email, "password": self.password},
                              catch_response=True) as response:
            if response.status_code == 200:
                # Si tu API devuelve un token en el body, guárdalo.
                # Si usas cookies HttpOnly, el cliente las guardará automáticamente.
                # self.token = response.json().get("access_token")
                # self.client.headers.update({"Authorization": f"Bearer {self.token}"})
                response.success()
            else:
                response.failure(f"Login falló: {response.status_code} - {response.text}")

    @task(3)  # Peso 3: se ejecutará 3 veces más que las de peso 1
    def ver_categorias(self):
        """Consulta la lista de categorías."""
        with self.client.get("/api/v1/categorias?activas_solo=true", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error al ver categorías: {response.status_code}")

    @task(2)  # Peso 2
    def ver_lecciones_de_categoria(self):
        """Consulta las lecciones de una categoría (ej. ID 1)."""
        categoria_id = 1  # Puedes aleatorizar esto si tienes varias categorías
        with self.client.get(f"/api/v1/lecciones?categoria_id={categoria_id}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error al ver lecciones: {response.status_code}")

    @task(1)  # Peso 1
    def ver_clases_de_leccion(self):
        """Consulta las clases de una lección (ej. ID 6 para Alfabeto A-Z)."""
        leccion_id = 6  # El ID de tu lección "Alfabeto A-Z" según tu base de datos
        with self.client.get(f"/api/v1/clases/leccion/{leccion_id}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error al ver clases: {response.status_code}")

class WebsiteUser(HttpUser):
    """
    Define el usuario virtual. wait_time simula el tiempo que un usuario
    real pasa leyendo antes de hacer clic en otra cosa.
    """
    wait_time = between(1, 3)  # Espera entre 1 y 3 segundos entre tareas
    tasks = [SignaFreeUser]