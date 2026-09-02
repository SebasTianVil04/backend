import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from .config_tipo_senas import detectar_tipo_sena, obtener_config_sena

logger = logging.getLogger(__name__)

def validar_calidad_seña(frames: List[np.ndarray], sena_esperada: str) -> Dict[str, Any]:
    """
    Valida la calidad de los frames para una seña específica.
    
    Args:
        frames: Lista de frames numpy arrays
        sena_esperada: Nombre de la seña esperada
    
    Returns:
        Dict con resultados de validación
    """
    if not frames:
        return {
            'valido': False, 
            'error': 'No hay frames',
            'frames_totales': 0,
            'frames_recomendados': 0,
            'puntaje_calidad': 0.0
        }
    
    try:
        # Detectar tipo de seña y configuración
        tipo_sena = detectar_tipo_sena(sena_esperada)
        config = obtener_config_sena(sena_esperada)
        
        # Validar número de frames
        frames_suficientes = len(frames) >= config['num_frames_recomendado'] * 0.6
        
        # Métricas de calidad
        metricas = {
            'nitidez_promedio': 0.0,
            'brillo_promedio': 0.0,
            'contraste_promedio': 0.0,
            'estabilidad_promedio': 0.0,
            'frames_validos': 0
        }
        
        variaciones = []
        frames_validos = 0
        
        # Analizar cada frame
        for i, frame in enumerate(frames):
            try:
                if frame is None or frame.size == 0:
                    continue
                
                # Convertir a escala de grises para análisis
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                # Calcular nitidez (varianza de Laplacian)
                nitidez = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Calcular brillo
                brillo = np.mean(gray)
                
                # Calcular contraste (desviación estándar)
                contraste = np.std(gray)
                
                # Umbrales de calidad
                umbral_nitidez = 25.0
                umbral_brillo_min = 15
                umbral_brillo_max = 240
                
                # Verificar calidad básica del frame
                if (nitidez > umbral_nitidez and 
                    umbral_brillo_min <= brillo <= umbral_brillo_max):
                    
                    metricas['nitidez_promedio'] += nitidez
                    metricas['brillo_promedio'] += brillo
                    metricas['contraste_promedio'] += contraste
                    frames_validos += 1
                
                # Calcular estabilidad entre frames consecutivos
                if i > 0 and i < min(10, len(frames)):
                    frame_actual = cv2.resize(gray, (100, 100))
                    frame_anterior = cv2.resize(
                        cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY) 
                        if len(frames[i-1].shape) == 3 else frames[i-1], 
                        (100, 100)
                    )
                    
                    # Calcular diferencia entre frames
                    diff = cv2.absdiff(frame_actual, frame_anterior)
                    mean_diff = diff.mean()
                    variaciones.append(mean_diff)
                    
            except Exception as e:
                logger.debug(f"Error analizando frame {i}: {e}")
                continue
        
        # Calcular promedios
        if frames_validos > 0:
            metricas['nitidez_promedio'] /= frames_validos
            metricas['brillo_promedio'] /= frames_validos
            metricas['contraste_promedio'] /= frames_validos
            metricas['frames_validos'] = frames_validos
        
        # Calcular estabilidad
        estabilidad_ok = True
        if variaciones:
            estabilidad_promedio = sum(variaciones) / len(variaciones)
            metricas['estabilidad_promedio'] = estabilidad_promedio
            
            # Umbral de estabilidad
            if config['requiere_estabilidad']:
                estabilidad_ok = estabilidad_promedio < 35
        
        # Calcular puntaje general de calidad
        puntaje_calidad = 0.0
        factores = []
        
        # Factor de frames suficientes (30%)
        factor_frames = min(1.0, len(frames) / config['num_frames_recomendado'])
        factores.append(('frames', factor_frames))
        
        # Factor de nitidez (25%)
        if metricas['nitidez_promedio'] > 0:
            factor_nitidez = min(1.0, metricas['nitidez_promedio'] / 100.0)
            factores.append(('nitidez', factor_nitidez))
        
        # Factor de brillo (20%)
        if metricas['brillo_promedio'] > 0:
            brillo_ideal = 128
            factor_brillo = 1.0 - abs(metricas['brillo_promedio'] - brillo_ideal) / brillo_ideal
            factor_brillo = max(0.0, min(1.0, factor_brillo))
            factores.append(('brillo', factor_brillo))
        
        # Factor de estabilidad (25%)
        factor_estabilidad = 1.0 if estabilidad_ok else 0.6
        factores.append(('estabilidad', factor_estabilidad))
        
        # Calcular puntaje ponderado
        pesos = {'frames': 0.3, 'nitidez': 0.25, 'brillo': 0.2, 'estabilidad': 0.25}
        for factor, valor in factores:
            puntaje_calidad += valor * pesos.get(factor, 0.0)
        
        # Determinar si es válido
        valido = (frames_suficientes and 
                 frames_validos >= len(frames) * 0.7 and
                 puntaje_calidad >= 0.5)
        
        resultado = {
            'valido': valido,
            'tipo_sena': tipo_sena,
            'puntaje_calidad': round(puntaje_calidad, 3),
            'frames_totales': len(frames),
            'frames_validos': frames_validos,
            'frames_recomendados': config['num_frames_recomendado'],
            'frames_suficientes': frames_suficientes,
            'estabilidad_ok': estabilidad_ok,
            'metricas': {
                'nitidez_promedio': round(metricas['nitidez_promedio'], 2),
                'brillo_promedio': round(metricas['brillo_promedio'], 2),
                'contraste_promedio': round(metricas['contraste_promedio'], 2),
                'estabilidad_promedio': round(metricas['estabilidad_promedio'], 2)
            },
            'factores_calidad': dict(factores)
        }
        
        # Añadir recomendaciones si la calidad es baja
        if not valido:
            recomendaciones = []
            
            if not frames_suficientes:
                recomendaciones.append(
                    f"Grabar más frames (actual: {len(frames)}, recomendado: {config['num_frames_recomendado']})"
                )
            
            if metricas['nitidez_promedio'] < 30:
                recomendaciones.append("Mejorar enfoque y evitar movimiento brusco")
            
            if metricas['brillo_promedio'] < 30 or metricas['brillo_promedio'] > 200:
                recomendaciones.append("Ajustar iluminación (muy oscuro o muy brillante)")
            
            if not estabilidad_ok:
                recomendaciones.append("Mantener la cámara más estable")
            
            resultado['recomendaciones'] = recomendaciones
        
        logger.info(f"Validación '{sena_esperada}': {valido}, puntaje: {puntaje_calidad:.3f}")
        return resultado
        
    except Exception as e:
        logger.error(f"Error en validación de calidad: {str(e)}")
        return {
            'valido': False,
            'error': f"Error en validación: {str(e)}",
            'frames_totales': len(frames) if frames else 0,
            'frames_recomendados': 0,
            'puntaje_calidad': 0.0
        }

def analizar_problemas_comunes(sena_esperada: str, sena_detectada: str, confianza: float) -> List[str]:
    """
    Analiza problemas comunes basado en las señas confundidas.
    """
    from .config_tipo_senas import es_confusion_comun, obtener_confusiones_comunes
    
    problemas = []
    
    # Verificar si es una confusión común
    if es_confusion_comun(sena_esperada, sena_detectada):
        problemas.append(f"Confusión común: '{sena_esperada}' se suele confundir con '{sena_detectada}'")
    
    # Verificar confianza baja
    if confianza < 0.4:
        problemas.append("Confianza muy baja en la detección")
    elif confianza < 0.6:
        problemas.append("Confianza moderada, podría mejorar")
    
    # Verificar tipos de seña
    tipo_esperado = detectar_tipo_sena(sena_esperada)
    tipo_detectado = detectar_tipo_sena(sena_detectada)
    
    if tipo_esperado != tipo_detectado:
        problemas.append(f"Inconsistencia en tipo: esperado {tipo_esperado}, detectado {tipo_detectado}")
    
    return problemas