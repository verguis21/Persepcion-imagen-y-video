"""Descriptores HOG + LBP con imagenes sinteticas."""
import numpy as np

from app.modulo_imagen.postura.descriptores import LONGITUD_VECTOR, recorte_persona, vector_hog_lbp


def imagen():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[50:250, 150:220] = 255
    return img


def test_recorte_valido_y_vector():
    recorte = recorte_persona(imagen(), [140, 40, 230, 260])
    assert recorte.shape == (128, 64) and recorte.dtype == np.uint8
    v = vector_hog_lbp(recorte)
    assert v.shape == (LONGITUD_VECTOR,) == (3790,)
    assert v.dtype == np.float32 and np.all(np.isfinite(v))


def test_caja_degenerada_o_fuera():
    assert recorte_persona(imagen(), [100, 50, 100, 200]) is None       # ancho cero
    assert recorte_persona(imagen(), [500, 400, 600, 500]) is None      # fuera de la imagen
    assert recorte_persona(imagen(), None) is None


def test_acepta_tupla_y_caja_que_sobresale():
    assert recorte_persona(imagen(), (-20, -20, 120, 330)).shape == (128, 64)
