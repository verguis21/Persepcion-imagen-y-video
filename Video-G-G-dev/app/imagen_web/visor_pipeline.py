"""Visor simple del pipeline del modulo Imagen: un boton por cada etapa ya programada.

Uso, desde la carpeta Video-G-G-dev y con el entorno activo:
    python -m app.imagen_web.visor_pipeline
y abrir http://127.0.0.1:5050

Es una herramienta de estudio y demostracion. No guarda nada en disco, no toca
el servidor de Video y solo escucha en esta computadora.
"""
from __future__ import annotations

import argparse
import base64
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from ..modulo_imagen.salida.evidencia import construir_fotograma
from ..modulo_imagen.calidad.nitidez import bloque_calidad
from ..modulo_imagen.calidad.preprocesamiento import mejorar_imagen
from ..modulo_imagen.calidad.restauracion import restaurar

MAX_BYTES = 12 * 1024 * 1024
# Calidad ficticia que dispara las dos operaciones de restaurar(): sirve para ver su efecto.
CALIDAD_FORZADA = {'ruido_mad_estimado': 1e9, 'nitidez_laplaciana': 0.0,
                   'desenfoque_movimiento': {'anisotropia': 0.0}}

app = Flask(__name__, template_folder='../templates')
app.config['MAX_CONTENT_LENGTH'] = MAX_BYTES
app.json.sort_keys = False  # conserva el orden de los campos del contrato


def _a_data_url(imagen, lado_max=1100):
    """JPEG en base64 para mostrar en la pagina (reducido si es muy grande)."""
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, lado_max / max(alto, ancho))
    if escala < 1:
        imagen = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode('.jpg', imagen, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError('No se pudo codificar la imagen')
    return 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode('ascii')


def _leer_imagen():
    archivo = request.files.get('imagen')
    if archivo is None:
        raise ValueError('Falta la imagen')
    frame = cv2.imdecode(np.frombuffer(archivo.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('El archivo no es una imagen JPG o PNG valida')
    return frame


def _restaurada(frame, forzar):
    """Cadena completa hasta la restauracion: (mejorada, restaurada, calidad, bloque)."""
    calidad = bloque_calidad(frame)
    mejorada, meta = mejorar_imagen(frame, modo='adaptativo')
    restaurada, bloque = restaurar(mejorada, CALIDAD_FORZADA if forzar else calidad)
    return mejorada, meta, restaurada, calidad, bloque


def _espectro(frame):
    """Imagen del espectro de Fourier (log de la magnitud, frecuencia cero al centro)."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitud = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gris - gris.mean()))))
    magnitud = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(magnitud, cv2.COLORMAP_INFERNO)


def paso_calidad(frame, forzar):
    return {'datos': bloque_calidad(frame),
            'nota': 'Medidas, no veredictos. apto_para_detalle queda vacio hasta elegir el umbral con datos propios.'}


def paso_fourier(frame, forzar):
    return {'imagen': _a_data_url(_espectro(frame)),
            'datos': {'energia_alta_frecuencia': bloque_calidad(frame)['energia_alta_frecuencia']},
            'nota': 'Centro = cambios lentos. Hacia afuera = bordes y detalle fino. Una foto desenfocada se apaga hacia afuera.'}


def paso_mejora(frame, forzar):
    mejorada, meta = mejorar_imagen(frame, modo='adaptativo')
    return {'imagen': _a_data_url(mejorada),
            'datos': {'operaciones': meta['operaciones'] or ['ninguna (la imagen ya estaba bien)'],
                      'luminosidad': [meta['antes']['luminosidad'], meta['despues']['luminosidad']],
                      'contraste': [meta['antes']['contraste'], meta['despues']['contraste']]},
            'nota': 'Mejora que ya existia: gamma, CLAHE y mediana. Los pares son [antes, despues].'}


def paso_restauracion(frame, forzar):
    _, _, restaurada, _, bloque = _restaurada(frame, forzar)
    datos = dict(bloque)
    datos['operaciones'] = bloque['operaciones'] or ['ninguna (las medidas no lo pidieron)']
    return {'imagen': _a_data_url(restaurada), 'datos': datos,
            'nota': 'Quita ruido y enfoca solo si hace falta. Marque "forzar" para ver el efecto en una foto buena.'}


def _analizar(frame):
    # Import diferido: YOLO se carga la primera vez que se pide este paso.
    from ..modulo_imagen.deteccion.analisis_tiempo_real import analizar_frame
    return analizar_frame(frame)


def paso_deteccion(frame, forzar):
    dato, anotada = _analizar(frame)
    personas = [{'postura_reglas_pose': p['estado_postural'],
                 'inclinacion_torso_grados': (p.get('pose') or {}).get('inclinacion_torso'),
                 'confianza_detector': p['confianza']} for p in dato['personas']]
    return {'imagen': _a_data_url(anotada),
            'datos': {'personas': personas,
                      'poses_sin_caja': len(dato['poses_sin_asociar']),
                      'celulares': len(dato['celulares']),
                      'otros_objetos': [o['clase'] for o in dato['otros_objetos']],
                      'advertencias': dato['advertencias'],
                      'tiempos_ms': dato['tiempos_ms']},
            'nota': 'Deteccion y pose con las redes que ya tenia el modulo. La postura sale de la regla por angulo.'}


def paso_evidencia(frame, forzar):
    dato, _ = _analizar(frame)
    _, _, _, calidad, bloque = _restaurada(frame, forzar)
    dato['segundo_s'] = 0.0
    dato['fotograma_indice'] = 0
    dato['evidencia_imagen'] = {'calidad': calidad, 'restauracion': bloque, 'deteccion_restaurada': None,
                                'escena': None, 'calibracion': None, 'posturas_clasificador': None}
    return {'datos': construir_fotograma(dato),
            'nota': 'Asi viaja un fotograma a Prediccion. Lo que aun no esta programado sale vacio (null, no_disponible, no_evaluable).'}


PASOS = {'calidad': paso_calidad, 'fourier': paso_fourier, 'mejora': paso_mejora,
         'restauracion': paso_restauracion, 'deteccion': paso_deteccion, 'evidencia': paso_evidencia}


@app.get('/')
def pagina():
    return render_template('pipeline_imagen.html')


@app.post('/api/paso/<nombre>')
def ejecutar_paso(nombre):
    if nombre not in PASOS:
        return jsonify({'error': 'Paso desconocido'}), 404
    try:
        frame = _leer_imagen()
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    forzar = request.form.get('forzar', '0') == '1'
    inicio = time.perf_counter()
    try:
        respuesta = PASOS[nombre](frame, forzar)
    except Exception:
        app.logger.exception('Fallo el paso %s', nombre)
        return jsonify({'error': 'El paso fallo; revise la consola del servidor'}), 500
    respuesta['ms'] = round((time.perf_counter() - inicio) * 1000)
    return jsonify(respuesta)


def main():
    parser = argparse.ArgumentParser(description='Visor del pipeline del modulo Imagen')
    parser.add_argument('--port', type=int, default=5050)
    args = parser.parse_args()
    print(f'Abra http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
