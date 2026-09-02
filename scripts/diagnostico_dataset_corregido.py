import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticadorDatasetCorregido:
    def __init__(self):
        self.metricas_validas = 0
        
    def analizar_video_corregido(self, ruta_video: str) -> Dict[str, any]:
        """Analiza un video con manejo robusto de errores."""
        cap = None
        try:
            if not os.path.exists(ruta_video):
                return {'error': 'Archivo no existe', 'valido': False}
            
            # Verificar tamaño del archivo
            file_size = os.path.getsize(ruta_video)
            if file_size < 1024:  # Menos de 1KB
                return {'error': 'Archivo demasiado pequeño', 'valido': False, 'tamaño': file_size}
            
            cap = cv2.VideoCapture(ruta_video)
            
            if not cap.isOpened():
                return {'error': 'No se pudo abrir con OpenCV', 'valido': False}
            
            # Obtener propiedades básicas con validación
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validaciones críticas
            if fps <= 0 or fps > 240:
                logger.warning(f"FPS inválido: {fps}, contando manualmente...")
                fps = self._calcular_fps_manual(cap)
            
            if total_frames <= 0:
                logger.warning(f"Frame count inválido: {total_frames}, contando manualmente...")
                total_frames = self._contar_frames_manual(cap)
            
            # Calcular duración de forma segura
            if fps > 0 and total_frames > 0:
                duracion = total_frames / fps
            else:
                duracion = 0
            
            # Leer algunos frames para verificar calidad
            calidades = []
            for i in range(min(10, total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret and frame is not None:
                    calidad = self._calcular_calidad_frame_seguro(frame)
                    calidades.append(calidad)
            
            calidad_promedio = np.mean(calidades) if calidades else 0
            
            # Determinar si es válido
            valido = (
                fps > 5 and 
                fps < 120 and 
                duracion > 0.5 and 
                duracion < 30 and 
                width >= 320 and 
                height >= 240 and
                calidad_promedio > 0.1
            )
            
            resultado = {
                'valido': valido,
                'fps': round(fps, 2),
                'duracion': round(duracion, 2),
                'resolucion': f"{width}x{height}",
                'total_frames': total_frames,
                'calidad_promedio': round(calidad_promedio, 3),
                'tamaño_archivo_mb': round(file_size / (1024*1024), 2),
                'problemas': self._identificar_problemas(fps, duracion, width, height, calidad_promedio)
            }
            
            if valido:
                self.metricas_validas += 1
            
            return resultado
            
        except Exception as e:
            logger.error(f"Error analizando {ruta_video}: {e}")
            return {'error': str(e), 'valido': False}
        finally:
            if cap is not None:
                cap.release()
    
    def _calcular_fps_manual(self, cap: cv2.VideoCapture) -> float:
        """Calcula FPS manualmente leyendo frames."""
        try:
            import time
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Leer primeros frames para calcular FPS
            start_time = time.time()
            frames_leidos = 0
            max_frames = 100
            
            for _ in range(max_frames):
                ret = cap.grab()
                if not ret:
                    break
                frames_leidos += 1
            
            end_time = time.time()
            tiempo_transcurrido = end_time - start_time
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if tiempo_transcurrido > 0:
                fps_calculado = frames_leidos / tiempo_transcurrido
                logger.info(f"FPS calculado manualmente: {fps_calculado:.1f}")
                return min(fps_calculado, 60)  # Limitar a 60 FPS máximo
            else:
                return 30.0  # Valor por defecto
                
        except Exception as e:
            logger.error(f"Error calculando FPS manual: {e}")
            return 30.0
    
    def _contar_frames_manual(self, cap: cv2.VideoCapture) -> int:
        """Cuenta frames manualmente."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_count = 0
            max_frames = 10000
            
            while frame_count < max_frames:
                ret = cap.grab()
                if not ret:
                    break
                frame_count += 1
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            logger.info(f"Frames contados manualmente: {frame_count}")
            return frame_count
            
        except Exception as e:
            logger.error(f"Error contando frames: {e}")
            return 0
    
    def _calcular_calidad_frame_seguro(self, frame: np.ndarray) -> float:
        """Calcula calidad de frame con manejo de errores."""
        try:
            if frame is None or frame.size == 0:
                return 0.0
            
            # Convertir a escala de grises
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Calcular nitidez
            nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calcular contraste
            contraste = np.std(gray)
            
            # Calcular brillo (ideal alrededor de 127)
            brillo = np.mean(gray)
            factor_brillo = 1.0 - min(abs(brillo - 127) / 127, 1.0)
            
            # Puntaje combinado normalizado
            puntaje_nitidez = min(nitidez / 100.0, 1.0)
            puntaje_contraste = min(contraste / 50.0, 1.0)
            
            puntaje_final = (
                puntaje_nitidez * 0.4 +
                puntaje_contraste * 0.3 + 
                factor_brillo * 0.3
            )
            
            return max(0.0, min(1.0, puntaje_final))
            
        except Exception as e:
            logger.warning(f"Error calculando calidad de frame: {e}")
            return 0.0
    
    def _identificar_problemas(self, fps: float, duracion: float, 
                             width: int, height: int, calidad: float) -> List[str]:
        """Identifica problemas específicos."""
        problemas = []
        
        if fps < 15:
            problemas.append(f"FPS bajo: {fps:.1f}")
        elif fps > 60:
            problemas.append(f"FPS muy alto: {fps:.1f}")
        
        if duracion < 1.0:
            problemas.append(f"Duración corta: {duracion:.1f}s")
        elif duracion > 10:
            problemas.append(f"Duración larga: {duracion:.1f}s")
        
        if width < 640 or height < 480:
            problemas.append(f"Resolución baja: {width}x{height}")
        
        if calidad < 0.3:
            problemas.append(f"Calidad pobre: {calidad:.2f}")
        elif calidad < 0.6:
            problemas.append(f"Calidad media: {calidad:.2f}")
        
        return problemas
    
    def analizar_dataset_completo(self, directorio_dataset: str) -> Dict[str, any]:
        """Analiza todo el dataset."""
        directorio = Path(directorio_dataset)
        
        # Buscar videos en formatos comunes
        formatos = ['*.webm', '*.mp4', '*.avi', '*.mov', '*.mkv']
        videos = []
        for formato in formatos:
            videos.extend(directorio.glob(f"**/{formato}"))
        
        logger.info(f"Encontrados {len(videos)} videos en {directorio}")
        
        resultados = {}
        metricas_totales = {
            'videos_totales': len(videos),
            'videos_validos': 0,
            'videos_invalidos': 0,
            'fps_promedio': 0,
            'duracion_promedio': 0,
            'calidad_promedio': 0,
            'resoluciones': {},
            'problemas_comunes': []
        }
        
        todos_problemas = []
        
        for i, video_path in enumerate(videos):
            logger.info(f"Analizando {i+1}/{len(videos)}: {video_path.name}")
            
            resultado = self.analizar_video_corregido(str(video_path))
            resultados[str(video_path)] = resultado
            
            if resultado.get('valido', False):
                metricas_totales['videos_validos'] += 1
                metricas_totales['fps_promedio'] += resultado.get('fps', 0)
                metricas_totales['duracion_promedio'] += resultado.get('duracion', 0)
                metricas_totales['calidad_promedio'] += resultado.get('calidad_promedio', 0)
                
                # Contar resoluciones
                resolucion = resultado.get('resolucion', 'desconocida')
                metricas_totales['resoluciones'][resolucion] = \
                    metricas_totales['resoluciones'].get(resolucion, 0) + 1
            else:
                metricas_totales['videos_invalidos'] += 1
            
            # Recopilar problemas
            problemas = resultado.get('problemas', [])
            todos_problemas.extend(problemas)
        
        # Calcular promedios solo para videos válidos
        if metricas_totales['videos_validos'] > 0:
            n = metricas_totales['videos_validos']
            metricas_totales['fps_promedio'] = round(metricas_totales['fps_promedio'] / n, 2)
            metricas_totales['duracion_promedio'] = round(metricas_totales['duracion_promedio'] / n, 2)
            metricas_totales['calidad_promedio'] = round(metricas_totales['calidad_promedio'] / n, 3)
        
        # Problemas más comunes
        problemas_comunes = Counter(todos_problemas).most_common(10)
        metricas_totales['problemas_comunes'] = problemas_comunes
        
        return {
            'resultados_detallados': resultados,
            'resumen': metricas_totales,
            'recomendaciones': self._generar_recomendaciones_finales(metricas_totales)
        }
    
    def _generar_recomendaciones_finales(self, metricas: Dict) -> List[str]:
        """Genera recomendaciones específicas."""
        recomendaciones = []
        
        total_videos = metricas['videos_totales']
        videos_validos = metricas['videos_validos']
        
        porcentaje_validos = (videos_validos / total_videos * 100) if total_videos > 0 else 0
        
        if porcentaje_validos < 50:
            recomendaciones.append(f"🚨 CRÍTICO: Solo {porcentaje_validos:.1f}% de videos son válidos")
            recomendaciones.append("Necesitas regrabar la mayoría de los videos")
        elif porcentaje_validos < 80:
            recomendaciones.append(f"⚠️  ADVERTENCIA: Solo {porcentaje_validos:.1f}% de videos son válidos")
            recomendaciones.append("Considera regrabar los videos problemáticos")
        else:
            recomendaciones.append(f"✅ BUENO: {porcentaje_validos:.1f}% de videos son válidos")
        
        if metricas['fps_promedio'] < 20:
            recomendaciones.append(f"FPS promedio bajo: {metricas['fps_promedio']}. Objetivo: 25-30 FPS")
        
        if metricas['duracion_promedio'] < 2.0:
            recomendaciones.append(f"Duración promedio corta: {metricas['duracion_promedio']}s. Objetivo: 3-5s")
        
        if metricas['calidad_promedio'] < 0.6:
            recomendaciones.append(f"Calidad visual baja: {metricas['calidad_promedio']}. Mejorar iluminación y enfoque")
        
        # Recomendaciones específicas basadas en problemas comunes
        for problema, count in metricas['problemas_comunes'][:3]:
            recomendaciones.append(f"Problema frecuente: {problema} ({count} videos)")
        
        return recomendaciones

def ejecutar_diagnostico_corregido():
    """Ejecuta el diagnóstico corregido."""
    diagnosticador = DiagnosticadorDatasetCorregido()
    
    print("=" * 70)
    print("DIAGNÓSTICO CORREGIDO DEL DATASET")
    print("=" * 70)
    
    resultado = diagnosticador.analizar_dataset_completo("archivos_subidos/videos_dataset")
    
    resumen = resultado['resumen']
    recomendaciones = resultado['recomendaciones']
    
    print(f"📊 ESTADÍSTICAS GENERALES:")
    print(f"   Videos totales: {resumen['videos_totales']}")
    print(f"   Videos válidos: {resumen['videos_validos']}")
    print(f"   Videos inválidos: {resumen['videos_invalidos']}")
    print(f"   Tasa de validez: {(resumen['videos_validos']/resumen['videos_totales']*100):.1f}%")
    
    if resumen['videos_validos'] > 0:
        print(f"\n📈 MÉTRICAS DE VIDEOS VÁLIDOS:")
        print(f"   FPS promedio: {resumen['fps_promedio']}")
        print(f"   Duración promedio: {resumen['duracion_promedio']}s")
        print(f"   Calidad visual promedio: {resumen['calidad_promedio']}")
    
    print(f"\n🖼️  RESOLUCIONES:")
    for resol, count in resumen['resoluciones'].items():
        print(f"   {resol}: {count} videos")
    
    print(f"\n❌ PROBLEMAS MÁS COMUNES:")
    for problema, count in resumen['problemas_comunes']:
        print(f"   {problema}: {count} ocurrencias")
    
    print(f"\n💡 RECOMENDACIONES:")
    for i, rec in enumerate(recomendaciones, 1):
        print(f"   {i}. {rec}")
    
    print("=" * 70)
    
    # Mostrar algunos ejemplos de videos inválidos
    print("\n🔍 EJEMPLOS DE VIDEOS INVÁLIDOS:")
    ejemplos_invalidos = 0
    for ruta, analisis in resultado['resultados_detallados'].items():
        if not analisis.get('valido', False) and ejemplos_invalidos < 5:
            print(f"   - {Path(ruta).name}: {analisis.get('error', 'Problemas de calidad')}")
            ejemplos_invalidos += 1
    
    print("=" * 70)
    
    return resultado

if __name__ == "__main__":
    ejecutar_diagnostico_corregido()