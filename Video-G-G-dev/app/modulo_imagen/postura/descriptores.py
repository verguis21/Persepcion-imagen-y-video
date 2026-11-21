"""Descriptores del recorte de una persona (etapa 6 del plan, sesion 6).

HOG (histograma de gradientes orientados): divide el recorte en celdas de 8x8
y en cada una cuenta hacia donde apuntan los bordes. Describe la SILUETA: una
persona de pie tiene sobre todo bordes verticales; una echada, horizontales.
LBP (patrones binarios locales): compara cada pixel con sus 8 vecinos y
resume la TEXTURA del recorte en un histograma de 10 valores.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

TAMANO = (64, 128)          # ancho x alto, el tamano clasico de HOG para personas
LONGITUD_VECTOR = 3790      # 3780 de HOG + 10 de LBP


def recorte_persona(frame, bbox, tam=TAMANO):
    """Recorte en gris de 128 x 64 (uint8), o None si la caja no sirve.

    `bbox` puede ser lista o tupla [x1, y1, x2, y2]; se ajusta a la imagen.
    """
    if bbox is None or len(bbox) != 4:
        return None
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1, x2 = max(0, x1), min(ancho, x2)
    y1, y2 = max(0, y1), min(alto, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    region = frame[y1:y2, x1:x2]
    gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    return cv2.resize(gris, tam, interpolation=cv2.INTER_AREA).astype(np.uint8)


def vector_hog_lbp(recorte):
    """Vector float32 de 3790 valores: HOG (3780) seguido del histograma LBP (10)."""
    v_hog = hog(recorte, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),          # explicito: el valor por defecto da 6804 valores
                block_norm='L2-Hys', feature_vector=True)
    lbp = local_binary_pattern(recorte, P=8, R=1, method='uniform')   # valores 0..9
    histograma, _ = np.histogram(lbp, bins=10, range=(0, 10))
    histograma = histograma.astype(np.float64) / max(1, histograma.sum())
    return np.concatenate([v_hog, histograma]).astype(np.float32)
