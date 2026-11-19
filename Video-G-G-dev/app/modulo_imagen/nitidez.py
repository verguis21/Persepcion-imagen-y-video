"""Medidas de nitidez y de desenfoque de movimiento (etapa 2 del plan).

Son medidas, no veredictos: un fotograma borroso NO se descarta, se describe.
- `energia_alta_frecuencia`: sesion 9 (Fourier). Una imagen enfocada reparte
  energia hacia las frecuencias altas; al desenfocarse, la energia se concentra
  en el centro del espectro.
- `desenfoque_movimiento`: sesion 6 (gradientes). Un borron en una direccion
  borra los bordes perpendiculares a ella, asi que la energia del gradiente
  queda desequilibrada entre X e Y.
"""
import cv2
import numpy as np

from .calidad import medir_calidad, validar_bgr

LADO_MAXIMO = 640  # misma miniatura que usa calidad.py: medidas comparables y baratas


def _gris_reducido(frame):
    """Gris float32 en miniatura (lado mayor <= LADO_MAXIMO). No muta la entrada."""
    validar_bgr(frame)
    alto, ancho = frame.shape[:2]
    escala = min(1.0, LADO_MAXIMO / max(alto, ancho))
    if escala < 1:
        frame = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def energia_alta_frecuencia(frame, radio_rel=0.10):
    """Fraccion (0..1) de la energia del espectro que cae FUERA de un disco central.

    El disco tiene radio `radio_rel * min(alto, ancho)`. La componente continua
    (el brillo medio) no cuenta: se resta la media antes de transformar.
    Devuelve 0.0 si la imagen es uniforme.
    """
    gris = _gris_reducido(frame)
    gris = gris - gris.mean()                      # quita la componente continua
    espectro = np.fft.fftshift(np.fft.fft2(gris))   # frecuencia cero al centro
    energia = np.abs(espectro) ** 2
    total = float(energia.sum())
    if total <= 1e-9:
        return 0.0
    alto, ancho = gris.shape
    v, u = np.ogrid[:alto, :ancho]
    distancia = np.sqrt((v - alto // 2) ** 2 + (u - ancho // 2) ** 2)
    fuera = distancia > radio_rel * min(alto, ancho)
    return round(float(energia[fuera].sum() / total), 4)


def desenfoque_movimiento(frame):
    """Desequilibrio direccional de los bordes.

    anisotropia = |Ex - Ey| / (Ex + Ey), entre 0 (bordes en todas direcciones)
    y 1 (bordes en una sola direccion). `direccion_grados` (0..180) es la
    orientacion dominante del gradiente, calculada con el tensor de estructura.

    Limite conocido: una escena con muchos bordes paralelos (los propios
    escalones) tiene anisotropia alta aunque este nitida. La medida se
    interpreta comparando con otros fotogramas de la misma camara.
    """
    gris = _gris_reducido(frame)
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    ex = float((gx * gx).sum())
    ey = float((gy * gy).sum())
    if ex + ey <= 1e-9:
        return {'anisotropia': 0.0, 'direccion_grados': 0.0}
    exy = float((gx * gy).sum())
    angulo = 0.5 * np.degrees(np.arctan2(2.0 * exy, ex - ey))
    return {'anisotropia': round(abs(ex - ey) / (ex + ey), 4),
            'direccion_grados': round(float(angulo % 180.0), 1)}


def bloque_calidad(frame, *, umbral_nitidez=None):
    """Bloque `calidad` del contrato imagen-evidencia-1.0 para un fotograma.

    Reutiliza `medir_calidad` sin modificarla. `apto_para_detalle` es None
    mientras el equipo no haya elegido `umbral_nitidez` con sus etiquetas.
    """
    base = medir_calidad(frame)
    nitidez = base['nitidez_laplaciana']
    return {
        'luminosidad': base['luminosidad'],
        'contraste': base['contraste'],
        'nitidez_laplaciana': nitidez,
        'energia_alta_frecuencia': energia_alta_frecuencia(frame),
        'ruido_mad_estimado': base['ruido_mad_estimado'],
        'desenfoque_movimiento': desenfoque_movimiento(frame),
        'apto_para_detalle': None if umbral_nitidez is None else bool(nitidez >= umbral_nitidez),
    }
