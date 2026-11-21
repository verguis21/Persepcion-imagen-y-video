"""Relaciones persona-objeto-pasamanos (etapa 5 del plan, sesion 6).

Todo se mide con las munecas del esqueleto y se divide entre la altura de la
persona: la misma distancia en pixeles es poca para alguien cerca de la camara
y mucha para alguien lejos.

Limite declarado: la foto es plana. Una mano que pasa por delante del
pasamanos sin tocarlo cuenta como contacto.
"""
from __future__ import annotations

import math

# Indices COCO del esqueleto: 9 = muneca izquierda, 10 = muneca derecha (de la persona).
MUNECAS = (('izquierda', 9), ('derecha', 10))
CONFIANZA_MINIMA_PUNTO = 0.5
# Umbrales PROVISIONALES: se sustituiran por los elegidos con la columna
# `toca_pasamanos` y `objeto_en_mano` cuando haya fotos etiquetadas.
UMBRAL_OBJETO_REL = 0.18
UMBRAL_PASAMANOS_REL = 0.12


def munecas_visibles(persona):
    """{'izquierda': (x, y), 'derecha': (x, y)} solo con las munecas confiables.

    Una persona sin esqueleto (`pose` vacio) no tiene munecas visibles.
    """
    puntos = (persona.get('pose') or {}).get('puntos_corporales') or []
    visibles = {}
    for mano, indice in MUNECAS:
        if len(puntos) > indice and len(puntos[indice]) >= 3 and float(puntos[indice][2]) >= CONFIANZA_MINIMA_PUNTO:
            visibles[mano] = (float(puntos[indice][0]), float(puntos[indice][1]))
    return visibles


def altura_persona(persona):
    caja = persona.get('bbox') or []
    return max(1.0, float(caja[3]) - float(caja[1])) if len(caja) == 4 else 1.0


def _distancia_punto_segmento(punto, segmento):
    x1, y1, x2, y2 = (float(v) for v in segmento)
    dx, dy = x2 - x1, y2 - y1
    largo2 = dx * dx + dy * dy
    if largo2 == 0:
        return math.dist(punto, (x1, y1))
    t = max(0.0, min(1.0, ((punto[0] - x1) * dx + (punto[1] - y1) * dy) / largo2))
    return math.dist(punto, (x1 + t * dx, y1 + t * dy))


def objetos_en_mano(persona, objetos, *, umbral_rel=UMBRAL_OBJETO_REL):
    """Objetos (no personas) cuyo centro esta cerca de una muneca visible."""
    munecas = munecas_visibles(persona)
    if not munecas:
        return []
    altura = altura_persona(persona)
    salida = []
    for indice, objeto in enumerate(objetos):
        caja = objeto.get('bbox') or []
        if objeto.get('clase') == 'person' or len(caja) != 4:
            continue
        centro = ((float(caja[0]) + float(caja[2])) / 2.0, (float(caja[1]) + float(caja[3])) / 2.0)
        mano, distancia = min(((m, math.dist(centro, p) / altura) for m, p in munecas.items()), key=lambda par: par[1])
        if distancia <= umbral_rel:
            confianza = objeto.get('confianza_detector', objeto.get('confianza'))
            salida.append({'indice_objeto': indice, 'clase': objeto.get('clase'), 'mano': mano,
                           'confianza_detector': confianza, 'distancia_relativa': round(distancia, 3)})
    return salida


def contacto_pasamanos(persona, segmento, *, umbral_rel=UMBRAL_PASAMANOS_REL):
    """'si', 'no' o 'no_evaluable' segun la muneca mas cercana a CUALQUIER pasamanos.

    `segmento` puede ser un solo segmento [x1, y1, x2, y2] o una lista de ellos
    (escaleras con baranda central y baranda de pared).
    """
    from ..escena.escena import segmentos_pasamanos
    segmentos = segmentos_pasamanos(segmento)
    munecas = munecas_visibles(persona)
    if not segmentos or not munecas:
        return 'no_evaluable'
    altura = altura_persona(persona)
    menor = min(_distancia_punto_segmento(p, s) for p in munecas.values() for s in segmentos) / altura
    return 'si' if menor < umbral_rel else 'no'
