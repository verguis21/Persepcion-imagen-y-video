"""Analiza fotogramas de clips de incidentes sin tracking ni detección de accidentes.

Video conserva la responsabilidad de capturar, identificar la escalera y elegir el
intervalo de interés. La función es reutilizable sin Flask ni una cámara abierta.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from ..deteccion.analisis_tiempo_real import analizar_frame
from ..deteccion.recuperacion_roi import recuperar_en_roi
from ..deteccion.pose_roi import recuperar_pose_en_roi
from ..salida.senalizacion import agregar_senales, resumen_clip
from ..deteccion.continuidad_visual import ContinuidadVisual


def tiempos_muestreo(duracion_s: float, intervalo_s: float = 1.0, tiempos_interes=None):
    if duracion_s <= 0:
        raise ValueError('La duración debe ser positiva')
    if intervalo_s <= 0:
        raise ValueError('El intervalo debe ser positivo')
    if tiempos_interes is not None:
        valores = [float(t) for t in tiempos_interes]
        if any(t < 0 or t > duracion_s for t in valores):
            raise ValueError('Hay tiempos fuera del clip')
        return sorted(set(round(min(t, duracion_s), 3) for t in valores))
    salida = []
    t = 0.0
    while t < duracion_s:
        salida.append(round(t, 3))
        t += intervalo_s
    return salida or [0.0]


def _posible_roi_por_desaparicion(previos, presentes, dimensiones, segundo):
    """Una búsqueda de respaldo, sin asignar identidad ni diagnosticar caídas.

    Evita regiones gigantes o personas cortadas por el borde, y no confunde a
    la persona visible en primer plano con otra persona distante.
    """
    h, w = dimensiones[:2]
    for previo in previos:
        b = previo['bbox']
        x1, y1, x2, y2 = map(float, b)
        area = max(0, x2-x1) * max(0, y2-y1)
        if area < h*w*.001 or area > h*w*.22 or x1 < 4 or y1 < 4 or x2 > w-4 or y2 > h-4:
            continue
        if not (0 < segundo - previo['segundo_s'] <= .65):
            continue
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if any(_cercano_bbox(p.get('bbox'), b) for p in presentes):
            continue
        return previo
    return None


def _cercano_bbox(actual, previo):
    if not isinstance(actual, (list,tuple)) or len(actual) != 4: return False
    ancho=max(1,previo[2]-previo[0]); alto=max(1,previo[3]-previo[1])
    ax=(actual[0]+actual[2])/2; ay=(actual[1]+actual[3])/2
    px=(previo[0]+previo[2])/2; py=(previo[1]+previo[3])/2
    return abs(ax-px) <= max(45, ancho*1.5) and abs(ay-py) <= max(40,alto*1.4)


def analizar_clip(ruta_clip, *, modo='adaptativo', intervalo_s=1.0,
                  tiempos_interes=None, guardar_en=None, contexto_video=None,
                  max_duracion_s=10.5, enriquecer=None):
    """Devuelve (resumen JSON-serializable, fotogramas procesados).

    El clip se lee con OpenCV. `contexto_video` conserva metadatos del detector
    de escaleras de Video; nunca ejecuta RiskEngine ni su tracking.
    `enriquecer`, si se pasa, es una función `(original, dato) -> None` que
    añade las piezas nuevas (calidad, restauración, escena, postura) bajo
    `dato['evidencia_imagen']`; sin ella el comportamiento es el de siempre.
    """
    ruta = Path(ruta_clip)
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise ValueError(f'No se pudo abrir el clip: {ruta.name}')
    try:
        fps = float(captura.get(cv2.CAP_PROP_FPS))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError('No se puede determinar FPS/duración del video')
        duracion = total / fps
        if duracion > max_duracion_s:
            raise ValueError(f'El clip excede {max_duracion_s:g} segundos')
        tiempos = tiempos_muestreo(duracion, intervalo_s, tiempos_interes)
        carpeta = Path(guardar_en) if guardar_en else None
        if carpeta:
            carpeta.mkdir(parents=True, exist_ok=True)
        resultados = []
        imagenes = []
        ultimo_candidato = None
        perdidas_consecutivas = 0
        visibles_previos = []
        continuidad = ContinuidadVisual(persistencia_s=2.0)
        inicio = time.perf_counter()
        for i, segundo in enumerate(tiempos):
            numero = min(int(round(segundo * fps)), total - 1)
            captura.set(cv2.CAP_PROP_POS_FRAMES, numero)
            ok, original = captura.read()
            if not ok or original is None:
                resultados.append({'segundo_s': segundo, 'estado': 'no_decodificado'})
                continue
            # Pueden usarse tanto clips crudos como seleccionados por Video.
            inicio_frame = time.perf_counter()
            dato, anotada = analizar_frame(original, modo=modo)
            dato['segundo_s'] = round(numero / fps, 3)
            dato['fotograma_indice'] = numero
            # Si las poses y YOLO global fallan, una segunda inferencia enfocada
            # en la ultima region observada puede recuperar una persona acostada.
            actual = dato['segundo_s']
            roi_resultado = None
            personas_actuales = (dato.get('personas', []) + dato.get('poses_sin_asociar', []))
            hubo_indicio_previo = (ultimo_candidato is not None and
                0 < actual - ultimo_candidato['segundo_s'] <= 2.0)
            objetivo = None
            if hubo_indicio_previo:
                anterior = ultimo_candidato['bbox']
                if not any(_cercano_bbox(p.get('bbox'), anterior) for p in personas_actuales):
                    objetivo = ultimo_candidato
            elif visibles_previos:
                objetivo = _posible_roi_por_desaparicion(visibles_previos, personas_actuales,
                                                         original.shape, actual)
            if objetivo is not None:
                previo = objetivo['bbox']
                t_roi = time.perf_counter()
                roi_resultado = recuperar_en_roi(original, previo)
                dato['tiempos_ms']['roi_yolo'] = round((time.perf_counter()-t_roi)*1000,1)
                if roi_resultado is None:
                    t_pose_roi = time.perf_counter()
                    roi_resultado = recuperar_pose_en_roi(original, previo)
                    dato['tiempos_ms']['roi_pose'] = round((time.perf_counter()-t_pose_roi)*1000,1)
                dato['recuperacion_roi'] = roi_resultado or {'estado':'sin_deteccion_actual'}
                if roi_resultado:
                    b = roi_resultado['bbox']
                    x1,y1,x2,y2 = map(int,b)
                    es_pose = roi_resultado['origen'] == 'roi_pose'
                    color = (255,255,0) if es_pose else (255,0,255)
                    cv2.rectangle(anotada,(x1,y1),(x2,y2),color,2)
                    cv2.putText(anotada, 'POSE ROI - revisar' if es_pose else 'YOLO ROI - revisar',
                                (x1,max(20,y1-7)), cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
                    dato['advertencias'].append('ROI recuperada: evidencia visual sin identidad ni accidente confirmado.')
                elif not hubo_indicio_previo:
                    dato['revision_incidente'] = 'perdida_visual_no_concluyente'
                    dato['advertencias'].append('Una persona observada previamente no pudo localizarse; esto no demuestra una caida.')
            observaciones = [p for p in dato.get('personas', []) if p.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision')]
            observaciones += [p for p in dato.get('poses_sin_asociar', []) if p.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision')]
            if observaciones:
                ultimo_candidato = {'segundo_s':dato['segundo_s'], 'bbox':observaciones[0]['bbox']}
                perdidas_consecutivas = 0
                dato['revision_incidente'] = 'indicio_visual_sin_confirmar'
            elif roi_resultado:
                if hubo_indicio_previo or roi_resultado.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision'):
                    ultimo_candidato = {'segundo_s': dato['segundo_s'], 'bbox': roi_resultado['bbox']}
                    dato['revision_incidente'] = ('persona_recuperada_pose_roi_sin_confirmar_incidente' if
                        roi_resultado['origen'] == 'roi_pose' else
                        'persona_recuperada_roi_sin_confirmar_incidente')
                else:
                    dato['revision_incidente'] = 'recuperacion_visual_sin_indicio_postural'
            elif ultimo_candidato and not dato.get('poses_sin_asociar') and not any(p.get('estado_postural') in ('postura_horizontal_posible','inclinacion_para_revision') for p in dato.get('personas', [])):
                # Se informa una perdida durante un breve periodo, sin fingir deteccion.
                segundos_perdidos = dato['segundo_s'] - ultimo_candidato['segundo_s']
                if 0 < segundos_perdidos <= 2.0:
                    perdidas_consecutivas += 1
                    dato['revision_incidente'] = 'deteccion_perdida_tras_indicio'
                    dato['ultima_observacion_estimada'] = {**ultimo_candidato, 'es_deteccion_actual':False}
                    dato['advertencias'].append('Se perdio la observacion posterior a una postura inusual; no hay deteccion confirmada en este fotograma.')
                    cv2.putText(anotada, 'OBSERVACION PERDIDA - revisar', (12, max(22, anotada.shape[0]-18)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,140,255), 2)
                else:
                    ultimo_candidato = None
                    dato['revision_incidente'] = 'sin_evidencia_actual'
            else:
                dato.setdefault('revision_incidente', 'sin_indicio_suficiente')
            visibles_previos = [
                {'bbox':p['bbox'], 'segundo_s':actual}
                for p in personas_actuales if len(p.get('bbox', [])) == 4
            ]
            if roi_resultado:
                visibles_previos.append({'bbox':roi_resultado['bbox'], 'segundo_s':actual})
            continuidad.aplicar(dato)
            if (dato.get('continuidad_visual') or {}).get('estado') == 'seguimiento_indicio_previo':
                cv2.putText(anotada, 'INDICIO PREVIO - EN SEGUIMIENTO',
                            (12, max(22, anotada.shape[0] - 42)),
                            cv2.FONT_HERSHEY_SIMPLEX, .54, (0, 175, 255), 2)
            if enriquecer is not None:
                enriquecer(original, dato)
            agregar_senales(dato)
            dato['tiempos_ms']['total_frame_con_roi'] = round((time.perf_counter()-inicio_frame)*1000,1)
            dato.pop('imagen_anotada', None)
            if carpeta:
                cv2.imwrite(str(carpeta / f'frame_{i:03d}_original.jpg'), original)
                cv2.imwrite(str(carpeta / f'frame_{i:03d}_anotado.jpg'), anotada)
            resultados.append(dato)
            imagenes.append(anotada)
        return {
            'modulo': 'imagen', 'estado': 'ok', 'clip': ruta.name,
            'duracion_s': round(duracion, 3), 'fps_origen': round(fps, 3),
            'fotogramas_planificados': len(tiempos),
            'fotogramas_analizados': len(imagenes),
            'tiempo_total_ms': round((time.perf_counter() - inicio)*1000, 1),
            'contexto_video': contexto_video or {},
            'resumen_visual': resumen_clip(resultados),
            'resultados': resultados,
            'nota': 'No confirma caídas; la interpretación temporal pertenece a Video/Predicción.'
        }, imagenes
    finally:
        captura.release()
