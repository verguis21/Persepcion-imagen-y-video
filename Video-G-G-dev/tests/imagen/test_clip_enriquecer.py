"""La puerta `enriquecer` de analizar_clip, con inferencias simuladas (sin YOLO)."""
import importlib
import sys
from types import SimpleNamespace

import cv2
import numpy as np


def _modulo_clip(monkeypatch):
    monkeypatch.setitem(sys.modules, 'app.modulo_imagen.deteccion.analisis_tiempo_real',
                        SimpleNamespace(analizar_frame=lambda *a, **k: None))
    mod = importlib.import_module('app.modulo_imagen.entrada.analizador_clip')
    def fake(frame, modo='adaptativo'):
        return {'personas': [], 'poses_sin_asociar': [], 'celulares': [], 'otros_objetos': [],
                'advertencias': [], 'tiempos_ms': {}, 'estado': 'ok'}, frame.copy()
    monkeypatch.setattr(mod, 'analizar_frame', fake)
    monkeypatch.setattr(mod, 'recuperar_en_roi', lambda *_: None)
    monkeypatch.setattr(mod, 'recuperar_pose_en_roi', lambda *_: None)
    return mod


def _clip(tmp_path, cuadros=4):
    ruta = tmp_path / 'prueba.avi'
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*'MJPG'), 5., (240, 220))
    assert w.isOpened()
    for _ in range(cuadros):
        w.write(np.full((220, 240, 3), 90, dtype=np.uint8))
    w.release()
    return ruta


def test_enriquecer_recibe_cada_fotograma_original(monkeypatch, tmp_path):
    mod = _modulo_clip(monkeypatch)
    vistos = []
    def enriquecer(original, dato):
        vistos.append(original.shape)
        dato['evidencia_imagen'] = {'marca': len(vistos)}
    res, _ = mod.analizar_clip(_clip(tmp_path), intervalo_s=.2, enriquecer=enriquecer)
    assert len(vistos) == res['fotogramas_analizados'] == 4
    assert all(forma == (220, 240, 3) for forma in vistos)
    assert [r['evidencia_imagen']['marca'] for r in res['resultados']] == [1, 2, 3, 4]


def test_sin_enriquecer_todo_sigue_igual(monkeypatch, tmp_path):
    mod = _modulo_clip(monkeypatch)
    clip = _clip(tmp_path)
    sin, _ = mod.analizar_clip(clip, intervalo_s=.2)
    con, _ = mod.analizar_clip(clip, intervalo_s=.2, enriquecer=lambda original, dato: None)
    assert all('evidencia_imagen' not in r for r in sin['resultados'])
    assert sin['resumen_visual'] == con['resumen_visual']
    assert [sorted(r) for r in sin['resultados']] == [sorted(r) for r in con['resultados']]
