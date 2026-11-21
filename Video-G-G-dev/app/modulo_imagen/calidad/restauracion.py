"""Restauracion condicionada (etapa 3 del plan, sesiones 3, 4 y 10).

Se aplica DESPUES de `mejorar_imagen` (gamma/CLAHE). Solo actua cuando las
medidas de calidad lo justifican:
- ruido alto            -> medias no locales (filtro no lineal que respeta bordes)
- nitidez baja sin borron de movimiento fuerte -> mascara de enfoque
Realzar un borron direccional crearia bordes falsos, por eso se evita.
"""
import cv2

from ..calidad.calidad import medir_calidad, validar_bgr

# Umbrales PROVISIONALES: se ajustan con los histogramas de los datos propios
# (scripts/imagen/calidad/comparar_restauracion.py --histograma-nitidez).
UMBRAL_RUIDO = 4.0         # ruido_mad_estimado por encima -> quitar ruido
UMBRAL_NITIDEZ = 120.0     # nitidez_laplaciana por debajo -> enfocar
UMBRAL_ANISOTROPIA = 0.35  # anisotropia por encima -> posible borron: no enfocar

FUERZA_NLM = 6             # parametro h de medias no locales
CANTIDAD_ENFOQUE = 0.8     # peso de la mascara de enfoque
SIGMA_ENFOQUE = 1.5        # desenfoque gaussiano usado para construir la mascara

MODOS = ('ninguno', 'adaptativo')


def _mascara_enfoque(img, cantidad=CANTIDAD_ENFOQUE, sigma=SIGMA_ENFOQUE):
    """original + cantidad * (original - suavizada). Resalta lo que el suavizado quita."""
    suave = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + cantidad, suave, -cantidad, 0)


def restaurar(frame, calidad, *, modo='adaptativo'):
    """Devuelve (imagen BGR restaurada, bloque `restauracion`). No muta `frame`.

    `calidad` es el dict de `nitidez.bloque_calidad` (o uno con las mismas claves).
    """
    validar_bgr(frame)
    if modo not in MODOS:
        raise ValueError("Modo invalido: use 'ninguno' o 'adaptativo'")
    img = frame.copy()
    operaciones = []
    nitidez_antes = medir_calidad(frame)['nitidez_laplaciana']
    if modo == 'adaptativo':
        ruido = float(calidad.get('ruido_mad_estimado') or 0.0)
        nitidez = float(calidad.get('nitidez_laplaciana') or 0.0)
        anisotropia = float((calidad.get('desenfoque_movimiento') or {}).get('anisotropia') or 0.0)
        if ruido > UMBRAL_RUIDO:
            img = cv2.fastNlMeansDenoisingColored(img, None, FUERZA_NLM, FUERZA_NLM, 7, 21)
            operaciones.append(f'nlm_h={FUERZA_NLM}')
        if nitidez < UMBRAL_NITIDEZ and anisotropia <= UMBRAL_ANISOTROPIA:
            img = _mascara_enfoque(img)
            operaciones.append(f'mascara_enfoque={CANTIDAD_ENFOQUE}')
    nitidez_despues = medir_calidad(img)['nitidez_laplaciana'] if operaciones else nitidez_antes
    return img, {'operaciones': operaciones,
                 'nitidez_antes': nitidez_antes,
                 'nitidez_despues': nitidez_despues}
