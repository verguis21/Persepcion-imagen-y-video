"""Contrato v1 entre Video e Imagen: metadatos validados, sin inferencia ni tracking.

Las cajas se expresan en pixeles (xyxy) de la imagen ORIGINAL. El contexto
aportado por Video es declarativo; Imagen no lo trata como verdad confirmada.
"""
from __future__ import annotations

import json
import math

VERSION = 'imagen-video-1.0.0'
MAX_METADATA_BYTES = 64 * 1024


def _texto(valor, etiqueta, limite=128):
    if not isinstance(valor, str) or not valor.strip() or len(valor) > limite:
        raise ValueError(f'{etiqueta} debe ser texto no vacio de hasta {limite} caracteres')
    return valor.strip()


def _numero(valor, etiqueta):
    # bool es una subclase de int en Python, pero no es una medida valida.
    if isinstance(valor, bool) or not isinstance(valor, (float, int)) or not math.isfinite(valor):
        raise ValueError(f'{etiqueta} debe ser un numero finito')
    return float(valor)


def _bbox(caja, ancho, alto):
    if not isinstance(caja, (list, tuple)) or len(caja) != 4:
        raise ValueError('bbox_xyxy_px debe tener cuatro coordenadas')
    x1, y1, x2, y2 = [_numero(c, 'coordenada bbox') for c in caja]
    if not (0 <= x1 < x2 <= ancho and 0 <= y1 < y2 <= alto):
        raise ValueError('bbox_xyxy_px debe estar dentro de las dimensiones del fotograma original')
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def leer_metadatos(texto):
    """Acepta JSON textual con limite; no admite propiedades extras arbitrarias."""
    if isinstance(texto, bytes):
        if len(texto) > MAX_METADATA_BYTES:
            raise ValueError('Metadatos demasiado grandes')
        texto = texto.decode('utf-8')
    if not isinstance(texto, str) or len(texto.encode('utf-8')) > MAX_METADATA_BYTES:
        raise ValueError('Metadatos demasiado grandes o invalidos')
    try:
        datos = json.loads(texto)
    except (ValueError, TypeError) as exc:
        raise ValueError('Los metadatos deben ser un objeto JSON valido') from exc
    if not isinstance(datos, dict):
        raise ValueError('Los metadatos deben ser un objeto JSON')
    return datos


def normalizar_metadatos(datos, *, ancho=None, alto=None, tipo='fotograma'):
    """Valida identidad, reloj y contexto. Copia solo los campos del contrato."""
    if not isinstance(datos, dict):
        raise ValueError('Los metadatos deben ser un objeto')
    clave = 'frame_id' if tipo == 'fotograma' else 'clip_id'
    identificador = _texto(datos.get(clave), clave)
    camara = _texto(datos.get('camara_id', 'camara_no_definida'), 'camara_id')
    tiempo = _numero(datos.get('tiempo_s', 0), 'tiempo_s')
    if tiempo < 0:
        raise ValueError('tiempo_s no puede ser negativo')
    contexto = datos.get('contexto_video', {})
    if not isinstance(contexto, dict):
        raise ValueError('contexto_video debe ser un objeto')
    escaleras = contexto.get('escaleras', [])
    tracks = contexto.get('tracks', [])
    if not isinstance(escaleras, list) or len(escaleras) > 25:
        raise ValueError('escaleras debe ser una lista de hasta 25 elementos')
    if not isinstance(tracks, list) or len(tracks) > 100:
        raise ValueError('tracks debe ser una lista de hasta 100 elementos')
    escaleras_ok = []
    for e in escaleras:
        if not isinstance(e, dict):
            raise ValueError('Cada escalera debe ser un objeto')
        caja = e.get('bbox_xyxy_px')
        if ancho is not None and alto is not None:
            caja = _bbox(caja, ancho, alto)
        elif not isinstance(caja, (list, tuple)) or len(caja) != 4:
            raise ValueError('Escalera sin bbox_xyxy_px valido')
        else:
            caja = [_numero(x, 'coordenada bbox') for x in caja]
            if not (caja[0] < caja[2] and caja[1] < caja[3] and caja[0] >= 0 and caja[1] >= 0):
                raise ValueError('bbox de escalera invalido')
        confianza = e.get('confianza')
        if confianza is not None:
            confianza = _numero(confianza, 'confianza')
            if not 0 <= confianza <= 1:
                raise ValueError('confianza fuera de rango')
        escaleras_ok.append({'bbox_xyxy_px': caja, 'confianza': confianza})
    tracks_ok = []
    for t in tracks:
        if not isinstance(t, dict):
            raise ValueError('Cada track debe ser un objeto')
        track_id = t.get('track_id')
        if not isinstance(track_id, (str, int)) or isinstance(track_id, bool) or len(str(track_id)) > 64:
            raise ValueError('track_id invalido')
        t_ok = {'track_id': track_id}
        if t.get('bbox_xyxy_px') is not None:
            if ancho is not None and alto is not None:
                t_ok['bbox_xyxy_px'] = _bbox(t['bbox_xyxy_px'], ancho, alto)
            else:
                t_ok['bbox_xyxy_px'] = _bbox(t['bbox_xyxy_px'], float('inf'), float('inf'))
        tracks_ok.append(t_ok)
    riesgo = contexto.get('riesgo_reportado_por_video')
    if riesgo is not None and riesgo not in ('BAJO', 'MEDIO', 'ALTO', 'NEUTRO', 'CONTEXTO'):
        raise ValueError('riesgo_reportado_por_video no reconocido')
    return {
        clave: identificador,
        'camara_id': camara,
        'tiempo_s': round(tiempo, 3),
        'contexto_video': {
            'escaleras': escaleras_ok,
            'tracks': tracks_ok,
            'riesgo_reportado_por_video': riesgo,
        },
    }


def _caja_video(caja, ancho, alto):
    """Ignora cajas malformadas. No transforma coordenadas normalizadas."""
    try:
        return _bbox(caja, ancho, alto)
    except (ValueError, TypeError):
        return None


def contexto_desde_video(respuesta, *, frame_id, camara_id='camara_local', tiempo_s=0):
    """Adapta /api/detect de Video a metadatos listos para Imagen.

    Un track se copia SOLO si Video entrega track_id; nunca se inventa.
    Detectores e inferencia se mantienen del lado de Video.
    """
    if not isinstance(respuesta, dict):
        raise ValueError('Respuesta de Video invalida')
    tamanio = respuesta.get('size')
    if not isinstance(tamanio, (list, tuple)) or len(tamanio) != 2:
        raise ValueError('Video no incluyo size=[ancho,alto]')
    ancho, alto = tamanio
    if isinstance(ancho, bool) or isinstance(alto, bool) or not isinstance(ancho, int) or not isinstance(alto, int) or ancho < 1 or alto < 1:
        raise ValueError('Video entrego dimensiones invalidas')
    escaleras, tracks = [], []
    for d in respuesta.get('detections', []):
        if not isinstance(d, dict):
            continue
        caja = _caja_video(d.get('bbox'), ancho, alto)
        if caja is None:
            continue
        nombre = str(d.get('class_name', '')).casefold()
        if nombre == 'escalera' or d.get('class_id') == 3:
            conf = d.get('confidence')
            escaleras.append({'bbox_xyxy_px': caja, 'confianza': conf})
        if d.get('track_id') is not None:
            tracks.append({'track_id': d['track_id'], 'bbox_xyxy_px': caja})
    return normalizar_metadatos({
        'frame_id': frame_id, 'camara_id': camara_id, 'tiempo_s': tiempo_s,
        'contexto_video': {
            'escaleras': escaleras[:25], 'tracks': tracks[:100],
            'riesgo_reportado_por_video': respuesta.get('riesgo'),
        },
    }, ancho=ancho, alto=alto)
