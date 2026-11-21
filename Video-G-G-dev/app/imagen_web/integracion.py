"""Puente HTTP v1 entre Video e Imagen.

Blueprint separado: NO importa RiskEngine, no toca el tracking ni la interfaz
principal. Diseñado para pruebas en localhost, nunca para Internet sin auth.
"""
from __future__ import annotations

import base64
import json
import tempfile
import math
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from ..modulo_imagen.entrada.contrato_video import (
    VERSION, leer_metadatos, normalizar_metadatos,
)

integracion_bp = Blueprint('integracion_video_imagen', __name__)
MAX_FRAME_BYTES = 7 * 1024 * 1024
MAX_CLIP_BYTES = 50 * 1024 * 1024
MAX_PIXELES = 12_000_000
MODOS = ('original', 'adaptativo', 'clahe', 'gamma')


def _miniatura_b64(ruta, ancho_max=720):
    """Retorna imagen de prueba real; limitada de tamaño para la respuesta HTTP."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        return None
    alto, ancho = imagen.shape[:2]
    if ancho > ancho_max:
        imagen = cv2.resize(imagen, (ancho_max, max(1, round(alto*ancho_max/ancho))),
                            interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', imagen, [cv2.IMWRITE_JPEG_QUALITY, 66])
    return 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii') if ok else None


def _muestras_visuales(carpeta, resultados, max_muestras=30):
    indices = [i for i in range(len(resultados))
               if (carpeta/f'frame_{i:03d}_original.jpg').is_file()
               and (carpeta/f'frame_{i:03d}_anotado.jpg').is_file()]
    if len(indices) > max_muestras:
        posiciones = {round(k*(len(indices)-1)/(max_muestras-1)) for k in range(max_muestras)}
        indices = [indices[i] for i in sorted(posiciones)]
    salida = []
    for i in indices:
        raw = _miniatura_b64(carpeta/f'frame_{i:03d}_original.jpg')
        anotada = _miniatura_b64(carpeta/f'frame_{i:03d}_anotado.jpg')
        if raw and anotada:
            salida.append({'indice_resultado':i,
                           'segundo_s':resultados[i].get('segundo_s'),
                           'original':raw,'anotada':anotada})
    return salida


def _archivo(nombre, limite):
    archivo = request.files.get(nombre)
    if archivo is None:
        raise ValueError(f'Falta archivo multipart: {nombre}')
    contenido = archivo.read(limite + 1)
    if not contenido:
        raise ValueError(f'{nombre} esta vacio')
    if len(contenido) > limite:
        raise OverflowError(f'{nombre} supera el limite de {limite // 1024 // 1024} MB')
    return contenido


def _configuracion(tipo):
    metadatos = normalizar_metadatos(
        leer_metadatos(request.form.get('metadatos', '{}')), tipo=tipo
    )
    modo = request.form.get('modo', 'adaptativo')
    if modo not in MODOS:
        raise ValueError('Modo de procesamiento no valido')
    return metadatos, modo


def _error_conocido(exc):
    codigo = 413 if isinstance(exc, OverflowError) else 400
    return jsonify({'estado': 'error', 'mensaje': str(exc), 'schema_version': VERSION}), codigo


@integracion_bp.get('/api/imagen/v1/salud')
def salud():
    return jsonify({'estado': 'ok', 'schema_version': VERSION,
                    'descripcion': 'Puente activo; no comprueba disponibilidad del modelo'})


@integracion_bp.post('/api/imagen/v1/fotograma')
def recibir_fotograma():
    try:
        if request.content_length and request.content_length > MAX_FRAME_BYTES + 128 * 1024:
            raise OverflowError('Solicitud de fotograma demasiado grande')
        contenido = _archivo('frame', MAX_FRAME_BYTES)
        frame = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Fotograma JPEG/PNG invalido')
        h, w = frame.shape[:2]
        if h * w > MAX_PIXELES:
            raise OverflowError('Fotograma supera los 12 megapixeles')
        metadatos, modo = _configuracion('fotograma')
        metadatos = normalizar_metadatos(metadatos, ancho=w, alto=h)
        incluir_anotada = request.form.get('incluir_anotada', '0') in ('1', 'true', 'True')
    except (ValueError, OverflowError, UnicodeDecodeError) as exc:
        return _error_conocido(exc)
    try:
        # Import diferido: el healthcheck funciona incluso sin cargar YOLO.
        from ..modulo_imagen.deteccion.analisis_tiempo_real import analizar_frame
        datos, anotada = analizar_frame(frame, modo=modo)
        from ..modulo_imagen.salida.senalizacion import agregar_senales
        agregar_senales(datos)
        respuesta = {
            'schema_version': VERSION,
            'estado': 'ok', 'modulo': 'imagen',
            'identificacion': {k: metadatos[k] for k in ('frame_id', 'camara_id', 'tiempo_s')},
            'contexto_video': metadatos['contexto_video'],
            'resultado_imagen': datos,
            'interpretacion': 'Evidencia visual. No confirma accidentes ni asigna IDs de tracking.',
        }
        if incluir_anotada:
            ok, jpg = cv2.imencode('.jpg', anotada, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('No se pudo codificar la imagen anotada')
            respuesta['imagen_anotada'] = 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode('ascii')
        return jsonify(respuesta)
    except Exception:
        current_app.logger.exception('Error en analisis de fotograma por Imagen')
        return jsonify({'estado': 'error', 'schema_version': VERSION,
                        'mensaje': 'No fue posible analizar el fotograma'}), 500


def _leer_peticion_clip():
    """Validaciones comunes de las rutas que reciben un clip (/clip y /evidencia).

    Devuelve (contenido, metadatos, modo, intervalo, extension) o lanza
    ValueError / OverflowError / UnicodeDecodeError.
    """
    if request.content_length and request.content_length > MAX_CLIP_BYTES + 128 * 1024:
        raise OverflowError('Solicitud de clip demasiado grande')
    contenido = _archivo('clip', MAX_CLIP_BYTES)
    metadatos, modo = _configuracion('clip')
    intervalo = float(request.form.get('intervalo_s', '1.0'))
    if not math.isfinite(intervalo) or not (0.1 <= intervalo <= 5.0):
        raise ValueError('intervalo_s debe estar entre 0.1 y 5.0 segundos')
    extension = Path(request.files['clip'].filename or '').suffix.lower()
    if extension not in ('.mp4', '.avi', '.mov', '.mkv'):
        raise ValueError('El clip debe ser MP4, AVI, MOV o MKV')
    return contenido, metadatos, modo, intervalo, extension


@integracion_bp.post('/api/imagen/v1/clip')
def recibir_clip():
    try:
        contenido, metadatos, modo, intervalo, extension = _leer_peticion_clip()
        incluir_muestras = request.form.get('incluir_muestras', '0') in ('1','true','True')
    except (ValueError, OverflowError, UnicodeDecodeError) as exc:
        return _error_conocido(exc)
    ruta = None
    try:
        from ..modulo_imagen.entrada.analizador_clip import analizar_clip
        # El original permanece en Video; esta copia temporal se elimina.
        with tempfile.NamedTemporaryFile(suffix=extension, prefix='imagen_', delete=False) as f:
            f.write(contenido)
            ruta = Path(f.name)
        with tempfile.TemporaryDirectory(prefix='muestras_imagen_') as d:
            carpeta = Path(d)
            resultado, _ = analizar_clip(
                ruta, modo=modo, intervalo_s=intervalo,
                guardar_en=carpeta if incluir_muestras else None,
                contexto_video=metadatos['contexto_video'], max_duracion_s=10.5,
            )
            # Identificacion estable del fotograma en la secuencia.
            for r in resultado.get('resultados', []):
                if 'fotograma_indice' in r:
                    r['frame_id'] = f"{metadatos['clip_id']}:{r['fotograma_indice']}"
            respuesta = {
                'schema_version': VERSION,
                'estado': 'ok', 'modulo': 'imagen',
                'identificacion': {k: metadatos[k] for k in ('clip_id', 'camara_id', 'tiempo_s')},
                'resultado_clip': resultado,
                'interpretacion': 'Evidencias visuales para revisión; no confirma accidentes.',
            }
            if incluir_muestras:
                respuesta['vista_previa'] = _muestras_visuales(carpeta, resultado.get('resultados', []))
            return jsonify(respuesta)
    except ValueError as exc:
        return _error_conocido(exc)
    except Exception:
        current_app.logger.exception('Error en analisis de clip por Imagen')
        return jsonify({'estado': 'error', 'schema_version': VERSION,
                        'mensaje': 'No fue posible analizar el clip'}), 500
    finally:
        if ruta is not None:
            ruta.unlink(missing_ok=True)



@integracion_bp.post('/api/imagen/v1/evidencia')
def recibir_clip_evidencia():
    """Mismo formulario que /clip; responde el JSON imagen-evidencia-1.0 para Prediccion."""
    try:
        contenido, metadatos, modo, intervalo, extension = _leer_peticion_clip()
    except (ValueError, OverflowError, UnicodeDecodeError) as exc:
        return _error_conocido(exc)
    ruta = None
    try:
        from ..modulo_imagen.salida.evidencia import evidencia_de_clip
        from ..modulo_imagen.escena.escena import cargar_calibracion
        modelo, variante = None, 'original'
        try:
            # El clasificador propio es la pieza de C; si aun no esta entrenado, se sigue sin el.
            from ..modulo_imagen.postura.clasificador_postura import cargar_modelo, variante_del_modelo
            modelo = cargar_modelo()
            variante = variante_del_modelo()
        except ImportError:
            modelo = None
        with tempfile.NamedTemporaryFile(suffix=extension, prefix='imagen_', delete=False) as f:
            f.write(contenido)
            ruta = Path(f.name)
        evidencia = evidencia_de_clip(
            ruta, clip_id=metadatos['clip_id'], camara_id=metadatos['camara_id'],
            contexto_video=metadatos['contexto_video'], modo=modo, intervalo_s=intervalo,
            modelo_postura=modelo, calibracion=cargar_calibracion(metadatos['camara_id']),
            variante_postura=variante)
        return jsonify(evidencia)
    except ValueError as exc:
        return _error_conocido(exc)
    except Exception:
        current_app.logger.exception('Error al armar la evidencia del clip')
        return jsonify({'estado': 'error', 'schema_version': VERSION,
                        'mensaje': 'No fue posible analizar el clip'}), 500
    finally:
        if ruta is not None:
            ruta.unlink(missing_ok=True)
