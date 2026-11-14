"""Pruebas HTTP con el cliente de Flask, sin pesos YOLO (se simulan inferencias)."""
import io
import json
import sys
import types

import cv2
import numpy as np
import pytest

pytest.importorskip('flask')
from app.servidor_imagen_integrado import app


def _jpeg():
    ok, jpg = cv2.imencode('.jpg', np.zeros((100,120,3), dtype=np.uint8))
    assert ok
    return jpg.tobytes()


def _frame_data(meta=None, raw=None):
    meta = meta or {'frame_id':'A:12', 'camara_id':'cam-01', 'tiempo_s':0.4,
                    'contexto_video':{'escaleras':[{'bbox_xyxy_px':[1,2,60,70],'confianza':.8}]}}
    return {'frame':(io.BytesIO(_jpeg() if raw is None else raw),'prueba.jpg'),
            'metadatos':json.dumps(meta)}


@pytest.fixture
def web():
    app.config['TESTING'] = True
    return app.test_client()


def test_salud_no_requiere_modelos(web):
    resp = web.get('/api/imagen/v1/salud')
    assert resp.status_code == 200
    assert resp.json['schema_version'] == 'imagen-video-1.0.0'


def test_fotograma_e2e_sin_yolo(web, monkeypatch):
    modulo = types.ModuleType('app.modulo_imagen.analisis_tiempo_real')
    modulo.analizar_frame = lambda frame, modo: (
        {'estado':'ok','personas':[{'bbox':[1,2,3,4]}],'advertencias':[]}, frame.copy())
    monkeypatch.setitem(sys.modules, modulo.__name__, modulo)
    resp = web.post('/api/imagen/v1/fotograma',data=_frame_data(),content_type='multipart/form-data')
    assert resp.status_code == 200
    assert resp.json['identificacion']['frame_id'] == 'A:12'
    assert resp.json['contexto_video']['escaleras'][0]['bbox_xyxy_px'] == [1,2,60,70]
    assert 'imagen_anotada' not in resp.json
    assert resp.json['resultado_imagen']['estado'] == 'ok'


def test_anotada_es_opcional(web, monkeypatch):
    modulo = types.ModuleType('app.modulo_imagen.analisis_tiempo_real')
    modulo.analizar_frame = lambda frame, modo: ({'personas':[]}, frame.copy())
    monkeypatch.setitem(sys.modules, modulo.__name__, modulo)
    data = _frame_data(); data['incluir_anotada'] = '1'
    resp = web.post('/api/imagen/v1/fotograma',data=data,content_type='multipart/form-data')
    assert resp.status_code == 200
    assert resp.json['imagen_anotada'].startswith('data:image/jpeg;base64,')


def test_rechazos_http(web):
    assert web.post('/api/imagen/v1/fotograma',data={}).status_code == 400
    assert web.post('/api/imagen/v1/fotograma',data=_frame_data(raw=b'basura'),content_type='multipart/form-data').status_code == 400
    invalido = _frame_data(meta={'frame_id':'X','contexto_video':{'escaleras':[{'bbox_xyxy_px':[0,0,1000,1000]}]}})
    assert web.post('/api/imagen/v1/fotograma',data=invalido,content_type='multipart/form-data').status_code == 400


def test_clip_temporal_se_borra(web, monkeypatch):
    module = types.ModuleType('app.modulo_imagen.analizador_clip')
    from pathlib import Path
    vistos=[]
    def simulado(ruta, **kwargs):
        ruta = Path(ruta)
        assert ruta.exists() and ruta.read_bytes() == b'contenido_ficticio'
        vistos.append(ruta)
        return ({'resultados':[{'fotograma_indice':8,'segundo_s':0.8}], 'estado':'ok'}, [])
    module.analizar_clip = simulado
    monkeypatch.setitem(sys.modules, module.__name__, module)
    resp=web.post('/api/imagen/v1/clip',data={
        'clip':(io.BytesIO(b'contenido_ficticio'),'incidente.mp4'),
        'metadatos':json.dumps({'clip_id':'grabacion_A','camara_id':'cam-01'}),
        'intervalo_s':'0.2',
    }, content_type='multipart/form-data')
    assert resp.status_code == 200
    assert resp.json['resultado_clip']['resultados'][0]['frame_id'] == 'grabacion_A:8'
    assert vistos and all(not p.exists() for p in vistos)
