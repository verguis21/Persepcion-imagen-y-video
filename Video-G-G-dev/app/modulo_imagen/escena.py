"""Descripcion de la escena (etapa 4 del plan, sesiones 4 y 5).

La camara es fija, asi que la escalera ocupa siempre los mismos pixeles. Una
persona marca una sola vez, con clics, la zona de escalones y el pasamanos
(scripts/imagen_calibrar_escena.py) y eso se guarda en
configs/escena/<camara_id>.json. Con esa zona:
- se sabe si un objeto o una persona esta sobre los escalones, y
- se cuentan los bordes de escalon visibles (Canny + morfologia + Hough), que
  es una medida de cuanta escalera se ve despejada.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np

CARPETA_CALIBRACION = Path('configs/escena')
TOLERANCIA_ANGULO = 12.0     # grados: lineas "de orientacion parecida"
LARGO_MINIMO_REL = 0.25      # una linea cuenta si mide al menos 25 % del ancho de la zona


def cargar_calibracion(camara_id, carpeta=CARPETA_CALIBRACION):
    """Lee configs/escena/<camara_id>.json. Devuelve None si no existe."""
    ruta = Path(carpeta) / f'{camara_id}.json'
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding='utf-8'))


def segmentos_pasamanos(valor):
    """Normaliza el campo `pasamanos` a una lista de segmentos [x1, y1, x2, y2].

    Acepta None, un solo segmento (formato anterior) o una lista de segmentos,
    para que las calibraciones antiguas sigan funcionando.
    """
    if not valor:
        return []
    if len(valor) == 4 and all(isinstance(v, (int, float)) for v in valor):
        return [list(valor)]
    return [list(s) for s in valor if s and len(s) == 4]


def escalar_calibracion(calibracion, ancho, alto):
    """Copia de la calibracion adaptada a un fotograma de otro tamano."""
    fx = ancho / calibracion['ancho_ref']
    fy = alto / calibracion['alto_ref']
    nueva = copy.deepcopy(calibracion)
    nueva['zona_escalones'] = [[int(round(x * fx)), int(round(y * fy))] for x, y in calibracion['zona_escalones']]
    segmentos = [[int(round(s[0] * fx)), int(round(s[1] * fy)), int(round(s[2] * fx)), int(round(s[3] * fy))]
                 for s in segmentos_pasamanos(calibracion.get('pasamanos'))]
    nueva['pasamanos'] = segmentos or None
    nueva['ancho_ref'], nueva['alto_ref'] = int(ancho), int(alto)
    return nueva


def punto_en_zona(punto, zona):
    """True si el punto (x, y) cae dentro del poligono o sobre su borde."""
    contorno = np.array(zona, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contorno, (float(punto[0]), float(punto[1])), False) >= 0


def _angulo(x1, y1, x2, y2):
    """Orientacion de un segmento en grados, entre 0 y 180."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _diferencia_angular(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def contar_lineas_escalon(frame, zona):
    """Cuenta bordes de escalon visibles dentro de la zona.

    1. Canny encuentra bordes.  2. Un cierre morfologico une bordes cortados.
    3. Se descarta lo que cae fuera del poligono.  4. Hough busca rectas.
    5. Se conservan las rectas largas con la orientacion dominante.
    6. Canny marca los dos lados de cada franja y Hough parte las rectas en
       trozos: los segmentos casi alineados se FUSIONAN antes de contar.
    """
    alto, ancho = frame.shape[:2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    bordes = cv2.Canny(cv2.GaussianBlur(gris, (5, 5), 0), 50, 150)
    bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    poligono = np.array(zona, dtype=np.int32)
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara, [poligono], 255)
    # Se encoge un poco la mascara para no contar el propio borde del poligono.
    mascara = cv2.erode(mascara, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    bordes = cv2.bitwise_and(bordes, mascara)
    x, y, ancho_zona, alto_zona = cv2.boundingRect(poligono)
    largo_minimo = max(20, int(LARGO_MINIMO_REL * ancho_zona))
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=30,
                             minLineLength=largo_minimo, maxLineGap=max(10, largo_minimo // 4))
    if lineas is None:
        return 0
    segmentos = [tuple(int(v) for v in linea[0]) for linea in lineas]
    # Orientacion dominante: la mediana ponderada por longitud es estable frente a rectas sueltas.
    datos = sorted((_angulo(*s), math.dist(s[:2], s[2:])) for s in segmentos)
    mitad, acumulado, dominante = sum(l for _, l in datos) / 2.0, 0.0, datos[0][0]
    for angulo, largo in datos:
        acumulado += largo
        if acumulado >= mitad:
            dominante = angulo
            break
    parecidos = [s for s in segmentos if _diferencia_angular(_angulo(*s), dominante) <= TOLERANCIA_ANGULO]
    if not parecidos:
        return 0
    # Posicion de cada segmento medida de forma perpendicular a la orientacion dominante.
    rad = math.radians(dominante)
    normal = (-math.sin(rad), math.cos(rad))
    posiciones = sorted(((s[0] + s[2]) / 2.0) * normal[0] + ((s[1] + s[3]) / 2.0) * normal[1] for s in parecidos)
    separacion = max(6.0, 0.02 * alto_zona)
    grupos, ultimo = 1, posiciones[0]
    for posicion in posiciones[1:]:
        if posicion - ultimo >= separacion:
            grupos += 1
        ultimo = posicion
    return grupos


def bloque_escena(frame, calibracion):
    """Bloque `escena` del contrato para un fotograma."""
    if calibracion is None:
        return {'origen_zona': 'no_disponible', 'zona_escalones': None,
                'lineas_escalon_detectadas': None, 'pasamanos': None}
    alto, ancho = frame.shape[:2]
    cal = escalar_calibracion(calibracion, ancho, alto)
    segmentos = segmentos_pasamanos(cal.get('pasamanos'))
    return {'origen_zona': 'calibracion_manual',
            'zona_escalones': cal['zona_escalones'],
            'lineas_escalon_detectadas': contar_lineas_escalon(frame, cal['zona_escalones']),
            # `visible` solo indica que hay pasamanos calibrado. `segmento` es el primero
            # (campo pactado en la version 1.0); `segmentos` trae todos.
            'pasamanos': {'visible': bool(segmentos), 'segmento': segmentos[0] if segmentos else None,
                          'segmentos': segmentos}}
