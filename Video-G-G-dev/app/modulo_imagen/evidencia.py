"""Contrato imagen-evidencia-1.0: evidencia medible para Predicción, sin veredictos.

Imagen describe lo que ve (calidad, escena, personas, objetos); no entrega
riesgo, probabilidad de accidente ni alertas. Este módulo arma y valida el JSON
a partir de lo que ya producen `analizar_frame` y `analizar_clip`.

Regla de importaciones: aquí NO se importan `detector`, `pose`,
`analisis_tiempo_real` ni `analizador_clip` en la cabecera, porque cargan YOLO
al importarse. Así las pruebas corren sin pesos ni descargas.
"""
from __future__ import annotations

import copy
from collections import Counter

VERSION = 'imagen-evidencia-1.0'
DETECCION_DOBLE = False   # A la cambia a True si el experimento de la etapa 3 muestra mejora
CLAVES_PROHIBIDAS = ('riesgo', 'probabilidad', 'alerta', 'accidente')

# Valores cerrados de la sección 4 del plan. Todos admiten además `None`.
VALORES_PERMITIDOS = {
    'postura_clasificador': ('erguida', 'inclinada', 'agachada_o_sentada', 'horizontal',
                             'no_evaluable', 'no_disponible'),
    'contacto_pasamanos': ('si', 'no', 'no_evaluable'),
    'origen_zona': ('calibracion_manual', 'no_disponible'),
    'mano': ('izquierda', 'derecha'),
    'postura_reglas_pose': ('sin_indicio_suficiente', 'inclinacion_para_revision',
                            'postura_horizontal_posible', 'no_evaluable'),
    'origen': ('yolo', 'solo_pose', 'roi_yolo', 'roi_pose'),
}

# Regla de pose (`_evaluar_postura`) -> clase equivalente del clasificador.
# La regla no tiene equivalente de `agachada_o_sentada`.
CORRESPONDENCIA_REGLA_CLASIFICADOR = {
    'sin_indicio_suficiente': 'erguida',
    'inclinacion_para_revision': 'inclinada',
    'postura_horizontal_posible': 'horizontal',
}

LIMITACIONES_BASE = (
    'Las posiciones y distancias se miden en proyección 2D sobre la imagen; '
    'no hay profundidad ni medidas reales en metros.',
    'Es evidencia descriptiva sin interpretación: Imagen no emite veredictos; '
    'la decisión corresponde a Predicción.',
)

_CLAVES_CALIDAD_ORIGINAL = ('luminosidad', 'contraste', 'nitidez_laplaciana', 'ruido_mad_estimado')


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------

def _revisar_claves(valor, ruta='evidencia'):
    """Recorre dicts y listas a cualquier profundidad buscando claves prohibidas."""
    if isinstance(valor, dict):
        for clave, contenido in valor.items():
            texto = str(clave).lower()
            for palabra in CLAVES_PROHIBIDAS:
                if palabra in texto:
                    raise ValueError(f'Clave prohibida {clave!r} en {ruta}: contiene {palabra!r}')
            _revisar_claves(contenido, f'{ruta}.{clave}')
    elif isinstance(valor, (list, tuple)):
        for i, elemento in enumerate(valor):
            _revisar_claves(elemento, f'{ruta}[{i}]')


def _revisar_valor(objeto, campo, ruta):
    if campo not in objeto:
        return
    valor = objeto[campo]
    if valor is not None and valor not in VALORES_PERMITIDOS[campo]:
        raise ValueError(f'{ruta}.{campo} tiene un valor no permitido: {valor!r}')


def _exigir(objeto, campo, ruta, tipo=None):
    if not isinstance(objeto, dict) or campo not in objeto:
        raise ValueError(f'Falta el campo obligatorio {ruta}.{campo}')
    valor = objeto[campo]
    if tipo is not None and not isinstance(valor, tipo):
        raise ValueError(f'{ruta}.{campo} debe ser {tipo.__name__}')
    return valor


def validar_evidencia(evidencia) -> None:
    """Lanza ValueError si falta un campo obligatorio, un valor cerrado no es
    válido o alguna clave contiene una palabra de CLAVES_PROHIBIDAS."""
    if not isinstance(evidencia, dict):
        raise ValueError('La evidencia debe ser un objeto')
    version = _exigir(evidencia, 'schema_version', 'evidencia')
    if version != VERSION:
        raise ValueError(f'schema_version debe ser {VERSION!r}, no {version!r}')
    clip = _exigir(evidencia, 'clip', 'evidencia', dict)
    clip_id = _exigir(clip, 'clip_id', 'clip')
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError('clip.clip_id debe ser texto no vacío')
    fotogramas = _exigir(evidencia, 'fotogramas', 'evidencia', list)
    for i, fotograma in enumerate(fotogramas):
        ruta = f'fotogramas[{i}]'
        if not isinstance(fotograma, dict):
            raise ValueError(f'{ruta} debe ser un objeto')
        _exigir(fotograma, 'segundo_s', ruta)
        calidad = _exigir(fotograma, 'calidad', ruta, dict)
        _exigir(calidad, 'nitidez_laplaciana', f'{ruta}.calidad')
        escena = fotograma.get('escena')
        if isinstance(escena, dict):
            _revisar_valor(escena, 'origen_zona', f'{ruta}.escena')
        personas = _exigir(fotograma, 'personas', ruta, list)
        for j, persona in enumerate(personas):
            ruta_p = f'{ruta}.personas[{j}]'
            if not isinstance(persona, dict):
                raise ValueError(f'{ruta_p} debe ser un objeto')
            _exigir(persona, 'bbox', ruta_p)
            _exigir(persona, 'postura_reglas_pose', ruta_p)
            en_mano = _exigir(persona, 'objetos_en_mano', ruta_p, list)
            for campo in ('postura_clasificador', 'contacto_pasamanos',
                          'postura_reglas_pose', 'origen'):
                _revisar_valor(persona, campo, ruta_p)
            for k, objeto in enumerate(en_mano):
                if not isinstance(objeto, dict):
                    raise ValueError(f'{ruta_p}.objetos_en_mano[{k}] debe ser un objeto')
                _revisar_valor(objeto, 'mano', f'{ruta_p}.objetos_en_mano[{k}]')
    _revisar_claves(evidencia)


# --------------------------------------------------------------------------
# Construcción
# --------------------------------------------------------------------------

def personas_de_dato(dato: dict) -> list[dict]:
    """Personas de un fotograma en orden fijo: YOLO, poses sin asociar y ROI.

    El paso 6 usa este mismo orden para alinear `posturas_clasificador`.
    """
    personas = []
    for p in dato.get('personas') or []:
        personas.append({'origen': 'yolo', 'detection_id': p.get('detection_id'),
                         'bbox': p.get('bbox'), 'confianza': p.get('confianza'),
                         'pose': p.get('pose'),
                         'estado_postural': p.get('estado_postural', 'no_evaluable')})
    for pose in dato.get('poses_sin_asociar') or []:
        personas.append({'origen': 'solo_pose', 'detection_id': None, 'bbox': pose['bbox'],
                         'confianza': None, 'pose': pose,
                         'estado_postural': pose['estado_postural']})
    roi = dato.get('recuperacion_roi')
    if isinstance(roi, dict) and roi.get('origen'):
        if roi['origen'] == 'roi_pose':
            personas.append({'origen': 'roi_pose', 'detection_id': None, 'bbox': roi.get('bbox'),
                             'confianza': None, 'pose': roi.get('pose'),
                             'estado_postural': roi.get('estado_postural', 'no_evaluable')})
        else:
            personas.append({'origen': roi['origen'], 'detection_id': None,
                             'bbox': roi.get('bbox'), 'confianza': roi.get('confianza'),
                             'pose': None,
                             'estado_postural': roi.get('estado_postural') or 'no_evaluable'})
    return personas


def concordancia_posturas(regla, clasificador):
    """True/False si ambos lados tienen equivalente; None en otro caso."""
    equivalente = CORRESPONDENCIA_REGLA_CLASIFICADOR.get(regla)
    if equivalente is None or clasificador not in CORRESPONDENCIA_REGLA_CLASIFICADOR.values():
        return None
    return equivalente == clasificador


def _lista(valor):
    return list(valor) if isinstance(valor, (list, tuple)) else valor


def _calidad_desde_original(dato):
    original = dato.get('calidad_original') or {}
    calidad = {clave: original.get(clave) for clave in _CLAVES_CALIDAD_ORIGINAL}
    calidad.update({'energia_alta_frecuencia': None, 'desenfoque_movimiento': None,
                    'apto_para_detalle': None})
    # Orden de claves igual al de la sección 4.
    orden = ('luminosidad', 'contraste', 'nitidez_laplaciana', 'energia_alta_frecuencia',
             'ruido_mad_estimado', 'desenfoque_movimiento', 'apto_para_detalle')
    return {clave: calidad[clave] for clave in orden}


def _base_de_caja(caja):
    """Punto medio de la base de una caja [x1, y1, x2, y2] (donde apoya en el suelo)."""
    return ((float(caja[0]) + float(caja[2])) / 2.0, float(caja[3]))


def _en_zona(caja, zona):
    if zona is None or caja is None or len(caja) != 4:
        return None
    from .escena import punto_en_zona
    return bool(punto_en_zona(_base_de_caja(caja), zona))


def construir_fotograma(dato: dict) -> dict:
    """Traduce un `dato` de `analizar_frame` al fotograma del contrato."""
    from .relaciones import contacto_pasamanos, objetos_en_mano
    extra = dato.get('evidencia_imagen') or {}
    posturas_clf = extra.get('posturas_clasificador')
    calibracion = extra.get('calibracion') or {}
    zona = calibracion.get('zona_escalones')
    segmento = calibracion.get('pasamanos')
    objetos = []
    for obj in list(dato.get('celulares') or []) + list(dato.get('otros_objetos') or []):
        caja = _lista(obj.get('bbox'))
        objetos.append({'clase': obj.get('clase'), 'bbox': caja,
                        'confianza_detector': obj.get('confianza'),
                        'en_zona_escalones': _en_zona(caja, zona), 'asociado_a_persona': None})
    crudas = personas_de_dato(dato)
    # Una mochila puesta cae "sobre los escalones" igual que una abandonada:
    # `solapa_persona` permite distinguirlas (centro del objeto dentro de la caja de alguien).
    for obj in objetos:
        caja = obj['bbox']
        centro = ((caja[0] + caja[2]) / 2.0, (caja[1] + caja[3]) / 2.0) if caja and len(caja) == 4 else None
        obj['solapa_persona'] = None if centro is None else any(
            len(p.get('bbox') or []) == 4 and p['bbox'][0] <= centro[0] <= p['bbox'][2]
            and p['bbox'][1] <= centro[1] <= p['bbox'][3] for p in crudas)
    # Candidatos de "objeto en mano" por persona; despues cada objeto se queda
    # solo con la persona mas cercana.
    candidatos = [objetos_en_mano(p, objetos) for p in crudas]
    mejor = {}
    for i, lista in enumerate(candidatos):
        for c in lista:
            actual = mejor.get(c['indice_objeto'])
            if actual is None or c['distancia_relativa'] < actual[1]:
                mejor[c['indice_objeto']] = (i, c['distancia_relativa'])
    personas = []
    for i, p in enumerate(crudas):
        regla = p['estado_postural']
        if posturas_clf is not None and i < len(posturas_clf):
            clasificador = posturas_clf[i]
        else:
            clasificador = 'no_disponible'
        pose = p.get('pose') or {}
        en_mano = []
        for c in candidatos[i]:
            if mejor[c['indice_objeto']][0] != i:
                continue
            objetos[c['indice_objeto']]['asociado_a_persona'] = p['detection_id']
            en_mano.append({'clase': c['clase'], 'mano': c['mano'], 'confianza_detector': c['confianza_detector']})
        personas.append({
            'detection_id': p['detection_id'],
            'origen': p['origen'],
            'bbox': _lista(p['bbox']),
            'confianza_detector': p['confianza'],
            'inclinacion_torso_grados': pose.get('inclinacion_torso'),
            'postura_reglas_pose': regla,
            'postura_clasificador': clasificador,
            'concordancia_posturas': concordancia_posturas(regla, clasificador),
            'en_zona_escalones': _en_zona(_lista(p['bbox']), zona),
            'contacto_pasamanos': contacto_pasamanos(p, segmento),
            'objetos_en_mano': en_mano,
        })
    calidad = extra.get('calidad')
    return {
        'indice': dato.get('fotograma_indice'),
        'segundo_s': dato.get('segundo_s'),
        'calidad': copy.deepcopy(calidad) if calidad is not None else _calidad_desde_original(dato),
        'restauracion': copy.deepcopy(extra.get('restauracion')),
        'deteccion_restaurada': copy.deepcopy(extra.get('deteccion_restaurada')),
        'escena': copy.deepcopy(extra.get('escena')),
        'personas': personas,
        'objetos': objetos,
    }


def _resumen(fotogramas):
    posturas = Counter()
    clases = Counter()
    for f in fotogramas:
        for p in f['personas']:
            if p['postura_clasificador'] not in (None, 'no_disponible', 'no_evaluable'):
                posturas[p['postura_clasificador']] += 1
        for o in f['objetos']:
            clases[o['clase']] += 1
    return {
        'fotogramas_analizados': len(fotogramas),
        'personas_max': max((len(f['personas']) for f in fotogramas), default=0),
        'fotogramas_con_desenfoque': sum(1 for f in fotogramas
                                         if (f.get('calidad') or {}).get('apto_para_detalle') is False),
        'posturas_observadas': dict(posturas),
        'clases_objeto': dict(clases),
    }


def construir_evidencia(resultado_clip: dict, *, clip_id: str, camara_id: str) -> dict:
    """Arma el JSON imagen-evidencia-1.0 a partir del resumen de `analizar_clip`.

    De `contexto_video` solo se copian `escaleras` y `tracks`; no se copian
    `resumen_visual`, `senales_visuales` ni `nota` (son interpretación).
    """
    contexto = resultado_clip.get('contexto_video') or {}
    fotogramas = [construir_fotograma(dato) for dato in resultado_clip.get('resultados') or []
                  if dato.get('estado') != 'no_decodificado']
    return {
        'schema_version': VERSION,
        'clip': {
            'clip_id': clip_id,
            'camara_id': camara_id,
            'duracion_s': resultado_clip.get('duracion_s'),
            'fps_origen': resultado_clip.get('fps_origen'),
            'contexto_video': {
                'escaleras': copy.deepcopy(list(contexto.get('escaleras') or [])),
                'tracks': copy.deepcopy(list(contexto.get('tracks') or [])),
            },
        },
        'fotogramas': fotogramas,
        'resumen': _resumen(fotogramas),
        'limitaciones': list(LIMITACIONES_BASE),
    }


def _iou(a, b):
    """Solapamiento entre dos cajas [x1, y1, x2, y2], entre 0 y 1."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def _deteccion_restaurada(restaurada, dato, hubo_operaciones, detector):
    """Segunda pasada del detector sobre la imagen restaurada, comparada con la normal."""
    normales = ([{'clase': 'person', 'bbox': p['bbox']} for p in dato.get('personas') or []]
                + list(dato.get('celulares') or []) + list(dato.get('otros_objetos') or []))
    if not hubo_operaciones:
        # La imagen no cambio: repetir la deteccion daria lo mismo.
        personas = sum(1 for d in normales if d['clase'] == 'person')
        return {'personas': personas, 'objetos': len(normales) - personas, 'solo_tras_restaurar': []}
    if detector is None:
        from .detector import detectar_imagen as detector   # carga YOLO: solo si hace falta
    detecciones = detector(restaurada)[1]
    nuevas = []
    for d in detecciones:
        if not any(n.get('clase') == d['clase'] and len(n.get('bbox') or []) == 4
                   and _iou(d['bbox'], n['bbox']) >= 0.5 for n in normales):
            nuevas.append({'clase': d['clase'], 'bbox': _lista(d['bbox']), 'confianza_detector': d.get('confianza')})
    personas = sum(1 for d in detecciones if d['clase'] == 'person')
    return {'personas': personas, 'objetos': len(detecciones) - personas, 'solo_tras_restaurar': nuevas}


def crear_enriquecedor(*, modelo_postura=None, calibracion=None, modo='adaptativo',
                       variante_postura='original', umbral_nitidez=None,
                       deteccion_doble=False, detector=None):
    """Devuelve la funcion `(original, dato) -> None` que se pasa a `analizar_clip`.

    Escribe SOLO `dato['evidencia_imagen']`; no toca ninguna otra clave.
    """
    from .escena import bloque_escena, escalar_calibracion
    from .nitidez import bloque_calidad
    from .preprocesamiento import mejorar_imagen
    from .restauracion import restaurar

    def enriquecer(original, dato):
        calidad = bloque_calidad(original, umbral_nitidez=umbral_nitidez)
        restaurada, restauracion = restaurar(mejorar_imagen(original, modo=modo)[0], calidad)
        alto, ancho = original.shape[:2]
        cal = escalar_calibracion(calibracion, ancho, alto) if calibracion else None
        personas = personas_de_dato(dato)
        if modelo_postura is None:
            posturas = ['no_disponible'] * len(personas)
        else:
            # El clasificador es la pieza de C (paso 5); se importa solo si hay modelo.
            from .clasificador_postura import predecir_postura
            variante = restaurada if variante_postura == 'restaurada' else original
            posturas = [predecir_postura(variante, p['bbox'], modelo_postura) for p in personas]
        dato['evidencia_imagen'] = {
            'calidad': calidad,
            'restauracion': restauracion,
            'deteccion_restaurada': (_deteccion_restaurada(restaurada, dato, bool(restauracion['operaciones']), detector)
                                     if deteccion_doble else None),
            'escena': bloque_escena(original, cal),
            'calibracion': cal,
            'posturas_clasificador': posturas,
        }
    return enriquecer


def evidencia_de_clip(ruta_clip, *, clip_id=None, camara_id='camara_local', contexto_video=None,
                      modo='adaptativo', intervalo_s=1.0, modelo_postura=None, calibracion=None,
                      deteccion_doble=None) -> dict:
    """Analiza un clip y devuelve el JSON imagen-evidencia-1.0 ya validado.

    No carga por su cuenta ni modelo ni calibracion: los pasa quien llama.
    `deteccion_doble=None` significa usar la constante DETECCION_DOBLE.
    """
    import json
    from pathlib import Path

    from .analizador_clip import analizar_clip   # carga YOLO: import diferido
    doble = DETECCION_DOBLE if deteccion_doble is None else bool(deteccion_doble)
    enriquecer = crear_enriquecedor(modelo_postura=modelo_postura, calibracion=calibracion,
                                    modo=modo, deteccion_doble=doble)
    resultado, _ = analizar_clip(ruta_clip, modo=modo, intervalo_s=intervalo_s,
                                 contexto_video=contexto_video, enriquecer=enriquecer)
    evidencia = construir_evidencia(resultado, clip_id=clip_id or Path(ruta_clip).stem, camara_id=camara_id)
    if calibracion is None:
        evidencia['limitaciones'].append(f"Sin calibración para la cámara '{camara_id}': "
                                         'zona de escalones y pasamanos no evaluados.')
    if modelo_postura is None:
        evidencia['limitaciones'].append('Clasificador de postura propio no disponible; solo regla de pose.')
    else:
        registro = Path('runs/imagen/postura_svm.json')
        detalle = ''
        if registro.is_file():
            try:
                info = json.loads(registro.read_text(encoding='utf-8'))
                detalle = f" con {info.get('filas')} filas de {info.get('clips')} clips propios"
            except (ValueError, OSError):
                detalle = ''
        evidencia['limitaciones'].append(f'Clasificador de postura entrenado{detalle}; ver informe de evaluación.')
    validar_evidencia(evidencia)
    return evidencia
