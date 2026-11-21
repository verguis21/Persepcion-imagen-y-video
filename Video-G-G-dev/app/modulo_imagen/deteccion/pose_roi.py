"""Rescate visual ROI mediante YOLO11-Pose, independiente del tracking de Video.

Se consulta SOLAMENTE cuando falla la detección global y el rescate ROI-YOLO.
Los puntos se convierten del recorte a coordenadas absolutas. No identifica
personas a través del tiempo ni confirma accidentes.
"""
from __future__ import annotations

import time

from ..deteccion.recuperacion_roi import crear_roi, _intersecta


def _puntos_visibles(pose, minimo_conf=0.35):
    valores = pose.get('puntos_corporales') or []
    return sum(1 for p in valores if len(p) >= 3 and p[2] >= minimo_conf)


def recuperar_pose_en_roi(frame, bbox_anterior, *, analizador_pose=None,
                         minimo_puntos=4, confianza_punto=0.35):
    """Devuelve una pose actual sustentada por puntos visibles, o None.

    `analizador_pose` es inyectable para pruebas; por defecto usa el modelo ya
    cargado desde modulo_imagen.deteccion.pose. La confianza de puntos NO se presenta
    como confianza de detección de persona.
    """
    if len(bbox_anterior) != 4:
        return None
    roi = crear_roi(bbox_anterior, frame.shape)
    x1, y1, x2, y2 = roi
    recorte = frame[y1:y2, x1:x2]
    if not recorte.size:
        return None
    if analizador_pose is None:
        from ..deteccion.pose import analizar_postura
        analizador_pose = analizar_postura
    inicio = time.perf_counter()
    _anotada, poses = analizador_pose(recorte)
    candidatos = []
    prev_cx = (bbox_anterior[0] + bbox_anterior[2]) / 2
    prev_cy = (bbox_anterior[1] + bbox_anterior[3]) / 2
    radio = max(65, max(bbox_anterior[2]-bbox_anterior[0],
                       bbox_anterior[3]-bbox_anterior[1]) * 0.90)
    for pose in poses:
        bbox = pose.get('bbox') or []
        if len(bbox) != 4 or _puntos_visibles(pose, confianza_punto) < minimo_puntos:
            continue
        global_bbox = [round(float(bbox[0]) + x1, 2), round(float(bbox[1]) + y1, 2),
                       round(float(bbox[2]) + x1, 2), round(float(bbox[3]) + y1, 2)]
        if global_bbox[2] <= global_bbox[0] or global_bbox[3] <= global_bbox[1]:
            continue
        cx = (global_bbox[0]+global_bbox[2]) / 2
        cy = (global_bbox[1]+global_bbox[3]) / 2
        cobertura = _intersecta(global_bbox, bbox_anterior)
        if cobertura < .1 and ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5 > radio:
            continue
        puntos = [
            [round(float(p[0]) + x1, 2), round(float(p[1]) + y1, 2), float(p[2])]
            for p in pose.get('puntos_corporales',[]) if len(p) >= 3
        ]
        if not puntos:
            continue
        visibles = [p[2] for p in puntos if p[2] >= confianza_punto]
        puntaje = len(visibles) + cobertura
        candidatos.append((puntaje, {**pose, 'bbox': global_bbox,
                         'puntos_corporales': puntos},len(visibles),cobertura))
    if not candidatos:
        return None
    _, pose, visibles, cobertura = max(candidatos, key=lambda v: v[0])
    # Import diferido para aprovechar las reglas de postura ya validadas.
    from ..deteccion.analisis_tiempo_real import _evaluar_postura
    return {
        'bbox':pose['bbox'],
        'origen':'roi_pose',
        'clase':'person',
        'pose':pose,
        'estado_postural':_evaluar_postura(pose),
        'puntos_visibles_cantidad':visibles,
        'solapamiento_caja_anterior':round(cobertura,3),
        'confianza':None,  # No se inventa confianza de caja YOLO.
        'roi_usada':list(roi),
        'tiempo_roi_ms':round((time.perf_counter()-inicio)*1000,1),
        'advertencia':'Hipótesis de pose en ROI; la identidad y el accidente no están confirmados.',
    }
