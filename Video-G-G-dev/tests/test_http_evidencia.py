"""Ruta POST /api/imagen/v1/evidencia con el cliente de Flask, sin YOLO."""
import io
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip('flask')
from app.servidor_imagen_integrado import app

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' / 'evidencia_ejemplo.json').read_text(encoding='utf-8'))


@pytest.fixture
def web():
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def evidencia_falsa(monkeypatch):
    vistos = []
    real = sys.modules.get('app.modulo_imagen.evidencia')
    falso = types.ModuleType('app.modulo_imagen.evidencia')
    def evidencia_de_clip(ruta, **kwargs):
        ruta = Path(ruta)
        assert ruta.exists()
        vistos.append((ruta, kwargs))
        salida = json.loads(json.dumps(FIXTURE))
        salida['clip']['clip_id'] = kwargs['clip_id']
        return salida
    falso.evidencia_de_clip = evidencia_de_clip
    monkeypatch.setitem(sys.modules, falso.__name__, falso)
    yield vistos
    if real is not None:
        sys.modules['app.modulo_imagen.evidencia'] = real


def _peticion(nombre='incidente.avi', **extra):
    datos = {'clip': (io.BytesIO(b'contenido_ficticio'), nombre),
             'metadatos': json.dumps({'clip_id': 'g2_1117_a1_celular_02', 'camara_id': 'escalera_g'})}
    datos.update(extra)
    return datos


def test_evidencia_responde_el_contrato_y_borra_el_temporal(web, evidencia_falsa):
    resp = web.post('/api/imagen/v1/evidencia', data=_peticion(), content_type='multipart/form-data')
    assert resp.status_code == 200
    assert resp.json['schema_version'] == 'imagen-evidencia-1.0'
    assert resp.json['clip']['clip_id'] == 'g2_1117_a1_celular_02'
    ruta, kwargs = evidencia_falsa[0]
    assert not ruta.exists()
    assert kwargs['camara_id'] == 'escalera_g' and kwargs['calibracion'] is None


def test_evidencia_rechaza_extension_e_intervalo(web, evidencia_falsa):
    assert web.post('/api/imagen/v1/evidencia', data=_peticion(nombre='nota.txt'),
                    content_type='multipart/form-data').status_code == 400
    assert web.post('/api/imagen/v1/evidencia', data=_peticion(intervalo_s='9'),
                    content_type='multipart/form-data').status_code == 400
    assert evidencia_falsa == []
