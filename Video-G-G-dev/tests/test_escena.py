"""Pruebas de escena con imagenes sinteticas; no cargan modelos."""
import json

import cv2
import numpy as np

from app.modulo_imagen.escena import (bloque_escena, cargar_calibracion, contar_lineas_escalon,
                                      escalar_calibracion, punto_en_zona)
from scripts.imagen_calibrar_escena import construir_calibracion

ZONA = [[40, 40], [360, 40], [360, 360], [40, 360]]


def lienzo():
    return np.zeros((400, 400, 3), dtype=np.uint8)


def test_seis_lineas_horizontales():
    img = lienzo()
    for y in range(70, 340, 48):                       # 6 lineas
        cv2.line(img, (60, y), (340, y), (255, 255, 255), 1)
    assert 5 <= contar_lineas_escalon(img, ZONA) <= 7


def test_una_franja_gruesa_cuenta_una_vez():
    img = lienzo()
    cv2.rectangle(img, (60, 198), (340, 201), (255, 255, 255), -1)   # 4 px de grosor
    assert contar_lineas_escalon(img, ZONA) == 1


def test_imagen_sin_bordes_da_cero():
    assert contar_lineas_escalon(lienzo(), ZONA) == 0


def test_punto_dentro_y_fuera():
    assert punto_en_zona((200, 200), ZONA) is True
    assert punto_en_zona((10, 10), ZONA) is False


def test_escalar_a_la_mitad():
    cal = {'camara_id': 'c', 'ancho_ref': 1280, 'alto_ref': 720,
           'zona_escalones': [[100, 200], [600, 200], [600, 700]], 'pasamanos': [640, 120, 800, 600]}
    media = escalar_calibracion(cal, 640, 360)
    assert media['zona_escalones'] == [[50, 100], [300, 100], [300, 350]]
    assert media['pasamanos'] == [[320, 60, 400, 300]]   # formato antiguo -> lista
    assert (media['ancho_ref'], media['alto_ref']) == (640, 360)
    assert cal['zona_escalones'][0] == [100, 200]      # no muta la original


def test_bloque_sin_calibracion():
    bloque = bloque_escena(lienzo(), None)
    assert bloque == {'origen_zona': 'no_disponible', 'zona_escalones': None,
                      'lineas_escalon_detectadas': None, 'pasamanos': None}


def test_bloque_con_calibracion_de_otro_tamano():
    cal = {'camara_id': 'c', 'ancho_ref': 800, 'alto_ref': 800,
           'zona_escalones': [[80, 80], [720, 80], [720, 720], [80, 720]], 'pasamanos': None}
    bloque = bloque_escena(lienzo(), cal)
    assert bloque['origen_zona'] == 'calibracion_manual'
    assert bloque['zona_escalones'] == ZONA
    assert bloque['pasamanos'] == {'visible': False, 'segmento': None, 'segmentos': []}
    assert bloque['lineas_escalon_detectadas'] == 0


def test_cargar_calibracion(tmp_path):
    assert cargar_calibracion('escalera_g', tmp_path) is None
    cal = construir_calibracion('escalera_g', 400, 400, [(40, 40), (360, 40), (360, 360)], [(10, 10), (90, 300)])
    (tmp_path / 'escalera_g.json').write_text(json.dumps(cal), encoding='utf-8')
    assert cargar_calibracion('escalera_g', tmp_path) == cal
    assert cal['pasamanos'] == [[10, 10, 90, 300]]


def test_calibracion_incompleta_se_rechaza():
    import pytest
    with pytest.raises(ValueError):
        construir_calibracion('c', 400, 400, [(1, 1), (2, 2)], [])
    with pytest.raises(ValueError):
        construir_calibracion('c', 400, 400, [(1, 1), (2, 2), (3, 9)], [(5, 5)])
    with pytest.raises(ValueError):
        construir_calibracion('c', 400, 400, [(1, 1), (2, 2), (3, 9)], [(5, 5), (9, 9), (20, 20)])


def test_varios_pasamanos_se_guardan_escalan_y_se_reportan():
    from app.modulo_imagen.escena import segmentos_pasamanos
    cal = construir_calibracion('c', 800, 800, [(80, 80), (720, 80), (720, 720), (80, 720)],
                                [(100, 0), (100, 800), (700, 0), (700, 800)])
    assert cal['pasamanos'] == [[100, 0, 100, 800], [700, 0, 700, 800]]
    bloque = bloque_escena(lienzo(), cal)
    assert bloque['pasamanos']['segmentos'] == [[50, 0, 50, 400], [350, 0, 350, 400]]
    assert bloque['pasamanos']['segmento'] == [50, 0, 50, 400] and bloque['pasamanos']['visible'] is True
    assert segmentos_pasamanos(None) == [] and segmentos_pasamanos([1, 2, 3, 4]) == [[1, 2, 3, 4]]
