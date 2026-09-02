import emails
import asyncio
from jinja2 import Template
from typing import Optional
from datetime import datetime
import pytz
from ..utilidades.configuracion import configuracion

class ServicioEmail:
    def __init__(self):
        self.smtp_options = {
            "host": configuracion.smtp_host,
            "port": configuracion.smtp_port,
            "tls": True,
            "user": configuracion.smtp_user,
            "password": configuracion.smtp_password,
        }
    
    async def enviar_email(self, 
                          destinatario: str, 
                          asunto: str, 
                          contenido_html: str,
                          contenido_texto: Optional[str] = None):
        """Enviar email de forma asíncrona"""
        try:
            message = emails.html(
                html=contenido_html,
                text=contenido_texto or "Versión HTML requerida",
                subject=asunto,
                mail_from=configuracion.mail_from
            )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: message.send(to=destinatario, smtp=self.smtp_options)
            )
            
            if hasattr(response, 'status_code') and response.status_code != 250:
                print(f"Error enviando email: {response.status_text}")
                return False
            
            print(f"Email enviado exitosamente a {destinatario}")
            return True
            
        except Exception as e:
            print(f"Error enviando email: {str(e)}")
            return False
    
    def enviar_email_sync(self, 
                         destinatario: str, 
                         asunto: str, 
                         contenido_html: str,
                         contenido_texto: Optional[str] = None):
        """Enviar email de forma síncrona (para background_tasks)"""
        try:
            message = emails.html(
                html=contenido_html,
                text=contenido_texto or "Versión HTML requerida",
                subject=asunto,
                mail_from=configuracion.mail_from
            )
            
            response = message.send(to=destinatario, smtp=self.smtp_options)
            
            if hasattr(response, 'status_code') and response.status_code != 250:
                print(f"Error enviando email: {getattr(response, 'status_text', 'Error desconocido')}")
                return False
            
            print(f"Email enviado exitosamente a {destinatario}")
            return True
            
        except Exception as e:
            print(f"Error enviando email: {str(e)}")
            return False
    
    def enviar_recuperacion_password(self, email: str, nombre: str, token: str):
        """Enviar email de recuperación de contraseña (versión síncrona para background_tasks)"""
        enlace_recuperacion = f"{configuracion.frontend_url}/restablecer-contrasena/{token}"
        
        template_html = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Recuperar Contraseña - SignaFree</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; }
                .logo { color: #3B82F6; font-size: 24px; font-weight: bold; }
                .button { display: inline-block; background-color: #3B82F6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; font-weight: bold; }
                .button:hover { background-color: #2563EB; }
                .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 12px; }
                .warning { background-color: #FEF3C7; padding: 15px; border-radius: 5px; margin: 20px 0; color: #92400E; border: 1px solid #FCD34D; }
                .code { word-break: break-all; color: #3B82F6; background: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">SignaFree</div>
                    <p style="color: #666; margin: 5px 0;">Aprende Lenguaje de Señas Peruano</p>
                </div>
                
                <h2 style="color: #1f2937; margin-bottom: 20px;">Recuperación de Contraseña</h2>
                
                <p>Hola <strong>{{ nombre }}</strong>,</p>
                
                <p>Recibimos una solicitud para restablecer la contraseña de tu cuenta en SignaFree.</p>
                
                <p>Si solicitaste este cambio, haz clic en el siguiente botón:</p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{{ enlace }}" class="button">Cambiar mi Contraseña</a>
                </div>
                
                <div class="warning">
                    <strong>Información importante:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Este enlace expira en <strong>30 minutos</strong></li>
                        <li>Solo puede usarse <strong>una sola vez</strong></li>
                        <li>Si no solicitaste este cambio, <strong>ignora este email</strong></li>
                    </ul>
                </div>
                
                <p>Si el botón no funciona, copia y pega este enlace en tu navegador:</p>
                <div class="code">{{ enlace }}</div>
                
                <p style="margin-top: 30px;">Si tienes problemas o preguntas, no dudes en contactarnos.</p>
                
                <p>Gracias por usar SignaFree</p>
                
                <div class="footer">
                    <p><strong>SignaFree</strong> - Plataforma de Aprendizaje de Lengua de Señas Peruano</p>
                    <p>Este es un email automático, no responder a esta dirección.</p>
                    <p>© 2025 SignaFree. Todos los derechos reservados.</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        contenido_html = template_html.render(
            nombre=nombre,
            enlace=enlace_recuperacion
        )
        
        contenido_texto = f"""
        SignaFree - Recuperación de Contraseña
        
        Hola {nombre},
        
        Recibimos una solicitud para restablecer la contraseña de tu cuenta en SignaFree.
        
        Para cambiar tu contraseña, visita el siguiente enlace:
        {enlace_recuperacion}
        
        IMPORTANTE:
        • Este enlace expira en 30 minutos
        • Solo puede usarse una vez
        • Si no solicitaste este cambio, ignora este email
        
        Si tienes problemas, contacta a nuestro soporte.
        
        Gracias por usar SignaFree
        
        ---
        SignaFree - Aprende Lenguaje de Señas Peruano
        Este es un email automático, no responder.
        © 2025 SignaFree
        """
        
        return self.enviar_email_sync(
            destinatario=email,
            asunto="Recupera tu contraseña - SignaFree",
            contenido_html=contenido_html,
            contenido_texto=contenido_texto
        )
    
    def enviar_notificacion_cambio_password(self, email: str, nombre: str):
        """Enviar email de notificación de cambio de contraseña exitoso"""
        
        template_html = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Contraseña Actualizada - SignaFree</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; }
                .logo { color: #10b981; font-size: 24px; font-weight: bold; }
                .success-icon { text-align: center; color: #10b981; font-size: 48px; margin: 20px 0; }
                .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 12px; }
                .info-box { background-color: #dbeafe; padding: 15px; border-radius: 5px; margin: 20px 0; color: #1e40af; border: 1px solid #93c5fd; }
                .warning-box { background-color: #fef3c7; padding: 15px; border-radius: 5px; margin: 20px 0; color: #92400e; border: 1px solid #fcd34d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">SignaFree</div>
                    <p style="color: #666; margin: 5px 0;">Aprende Lenguaje de Señas Peruano</p>
                </div>
                
                <div class="success-icon">✓</div>
                
                <h2 style="color: #1f2937; margin-bottom: 20px; text-align: center;">Contraseña Actualizada</h2>
                
                <p>Hola <strong>{{ nombre }}</strong>,</p>
                
                <p>Te confirmamos que tu contraseña ha sido <strong>actualizada exitosamente</strong>.</p>
                
                <div class="info-box">
                    <strong>Detalles del cambio:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li><strong>Fecha:</strong> {{ fecha }}</li>
                        <li><strong>Hora:</strong> {{ hora }}</li>
                        <li><strong>Cuenta:</strong> {{ email }}</li>
                    </ul>
                </div>
                
                <div class="warning-box">
                    <strong>¿No fuiste tú?</strong>
                    <p style="margin: 10px 0;">Si no realizaste este cambio, tu cuenta podría estar comprometida. Por favor:</p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Cambia tu contraseña inmediatamente</li>
                        <li>Contacta a nuestro equipo de soporte</li>
                        <li>Revisa la actividad reciente de tu cuenta</li>
                    </ul>
                </div>
                
                <p style="margin-top: 30px;">Si fuiste tú quien realizó el cambio, puedes ignorar este mensaje.</p>
                
                <p>Gracias por mantener tu cuenta segura</p>
                
                <div class="footer">
                    <p><strong>SignaFree</strong> - Plataforma de Aprendizaje de Lengua de Señas Peruano</p>
                    <p>Este es un email automático, no responder a esta dirección.</p>
                    <p>© 2025 SignaFree. Todos los derechos reservados.</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        # Usar zona horaria de Perú
        tz_peru = pytz.timezone('America/Lima')
        ahora = datetime.now(tz_peru)
        
        fecha_formateada = ahora.strftime('%d/%m/%Y')
        hora_formateada = ahora.strftime('%I:%M %p')
        
        contenido_html = template_html.render(
            nombre=nombre,
            email=email,
            fecha=fecha_formateada,
            hora=hora_formateada
        )
        
        contenido_texto = f"""
        SignaFree - Contraseña Actualizada
        
        Hola {nombre},
        
        Te confirmamos que tu contraseña ha sido actualizada exitosamente.
        
        Detalles del cambio:
        - Fecha: {fecha_formateada}
        - Hora: {hora_formateada}
        - Cuenta: {email}
        
        ¿No fuiste tú?
        Si no realizaste este cambio, tu cuenta podría estar comprometida:
        • Cambia tu contraseña inmediatamente
        • Contacta a nuestro equipo de soporte
        • Revisa la actividad reciente de tu cuenta
        
        Si fuiste tú quien realizó el cambio, puedes ignorar este mensaje.
        
        Gracias por mantener tu cuenta segura
        
        ---
        SignaFree - Aprende Lenguaje de Señas Peruano
        Este es un email automático, no responder.
        © 2025 SignaFree
        """
        
        return self.enviar_email_sync(
            destinatario=email,
            asunto="Contraseña actualizada - SignaFree",
            contenido_html=contenido_html,
            contenido_texto=contenido_texto
        )

# Instancia global
servicio_email = ServicioEmail()