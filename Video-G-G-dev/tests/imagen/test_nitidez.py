"""Pruebas de nitidez con imagenes sinteticas; no cargan modelos."""
import cv2
import numpy as np

from app.modulo_imagen.calidad.nitidez import (bloque_calidad, desenfoque_movimiento,
                                       energia_alta_frecuencia)


def tablero(lado=256, celda=16):
    y, x = np.indices((lado, lado))
    gris = (((x // celda) + (y // celda)) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def test_nitida_tiene_mas_alta_frecuencia_que_desenfocada():
    nitida = tablero()
    borrosa = cv2.GaussianBlur(nitida, (0, 0), 4)
    assert energia_alta_frecuencia(nitida) > energia_alta_frecuencia(borrosa)


def test_borron_horizontal_sube_la_anisotropia():
    original = tablero()
    nucleo = np.ones((1, 15), dtype=np.float32) / 15
    corrida = cv2.filter2D(original, -1, nucleo)
    assert desenfoque_movimiento(corrida)['anisotropia'] > desenfoque_movimiento(original)['anisotropia']


def test_imagen_uniforme_da_cero_sin_error():
    lisa = np.full((120, 160, 3), 128, dtype=np.uint8)
    assert energia_alta_frecuencia(lisa) == 0.0
    assert desenfoque_movimiento(lisa) == {'anisotropia': 0.0, 'direccion_grados': 0.0}


def test_bloque_calidad_sin_umbral_y_con_umbral():
    img = tablero()
    sin = bloque_calidad(img)
    assert sin['apto_para_detalle'] is None
    assert set(sin) == {'luminosidad', 'contraste', 'nitidez_laplaciana', 'energia_alta_frecuencia',
                        'ruido_mad_estimado', 'desenfoque_movimiento', 'apto_para_detalle'}
    assert bloque_calidad(img, umbral_nitidez=0.001)['apto_para_detalle'] is True
    assert bloque_calidad(img, umbral_nitidez=1e12)['apto_para_detalle'] is False


def test_no_muta_la_entrada():
    img = tablero()
    copia = img.copy()
    bloque_calidad(img)
    assert np.array_equal(img, copia)
