"""Pruebas de restauracion con imagenes sinteticas; no cargan modelos."""
import cv2
import numpy as np
import pytest

from app.modulo_imagen.calidad import restauracion
from app.modulo_imagen.calidad.restauracion import restaurar


def escena_limpia():
    img = np.full((160, 200, 3), 90, dtype=np.uint8)
    cv2.rectangle(img, (40, 30), (150, 130), (200, 200, 200), -1)
    cv2.circle(img, (95, 80), 25, (40, 40, 40), -1)
    return img


def test_quita_ruido_sin_mutar_la_entrada():
    limpia = escena_limpia()
    rng = np.random.default_rng(7)
    ruidosa = np.clip(limpia.astype(np.float32) + rng.normal(0, 18, limpia.shape), 0, 255).astype(np.uint8)
    copia = ruidosa.copy()
    calidad = {'ruido_mad_estimado': restauracion.UMBRAL_RUIDO * 5, 'nitidez_laplaciana': 1e6,
               'desenfoque_movimiento': {'anisotropia': 0.0}}
    salida, bloque = restaurar(ruidosa, calidad)
    error_antes = np.abs(ruidosa.astype(np.float32) - limpia).mean()
    error_despues = np.abs(salida.astype(np.float32) - limpia).mean()
    assert error_despues < error_antes
    assert np.array_equal(ruidosa, copia)
    assert any(op.startswith('nlm') for op in bloque['operaciones'])


def test_modo_ninguno_devuelve_copia_identica():
    img = escena_limpia()
    salida, bloque = restaurar(img, {'ruido_mad_estimado': 99, 'nitidez_laplaciana': 0}, modo='ninguno')
    assert np.array_equal(salida, img) and salida is not img
    assert bloque['operaciones'] == []
    assert bloque['nitidez_antes'] == bloque['nitidez_despues']


def test_no_enfoca_si_hay_borron_de_movimiento():
    calidad = {'ruido_mad_estimado': 0.0, 'nitidez_laplaciana': 1.0,
               'desenfoque_movimiento': {'anisotropia': restauracion.UMBRAL_ANISOTROPIA + 0.3}}
    _, bloque = restaurar(escena_limpia(), calidad)
    assert not any(op.startswith('mascara_enfoque') for op in bloque['operaciones'])


def test_enfoca_imagen_poco_nitida_y_sube_la_nitidez():
    borrosa = cv2.GaussianBlur(escena_limpia(), (0, 0), 3)
    calidad = {'ruido_mad_estimado': 0.0, 'nitidez_laplaciana': 1.0,
               'desenfoque_movimiento': {'anisotropia': 0.0}}
    _, bloque = restaurar(borrosa, calidad)
    assert bloque['operaciones'] == [f'mascara_enfoque={restauracion.CANTIDAD_ENFOQUE}']
    assert bloque['nitidez_despues'] > bloque['nitidez_antes']


def test_modo_desconocido():
    with pytest.raises(ValueError):
        restaurar(escena_limpia(), {}, modo='magia')
