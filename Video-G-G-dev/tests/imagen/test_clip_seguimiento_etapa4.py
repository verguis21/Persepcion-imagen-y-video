"""Comprueba señalización del analizador de clip con resultados simulados."""
import importlib
import sys
from types import SimpleNamespace
import cv2
import numpy as np


def test_clip_mantiene_aviso_solo_por_ventana_temporal(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules,'app.modulo_imagen.deteccion.analisis_tiempo_real',
                        SimpleNamespace(analizar_frame=lambda *args, **kwargs: None))
    mod=importlib.import_module('app.modulo_imagen.entrada.analizador_clip')
    n={'v':0}
    def fake(frame,modo='adaptativo'):
        n['v']+=1
        personas=[{'bbox':[50,50,120,190], 'estado_postural':'postura_horizontal_posible'}] if n['v']==1 else []
        return {'personas':personas,'poses_sin_asociar':[],'advertencias':[],
                'tiempos_ms':{}, 'estado':'ok'},frame.copy()
    monkeypatch.setattr(mod,'analizar_frame',fake)
    monkeypatch.setattr(mod,'recuperar_en_roi',lambda *_:None)
    monkeypatch.setattr(mod,'recuperar_pose_en_roi',lambda *_:None)
    clip=tmp_path/'test.avi'
    w=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),5.,(240,220))
    assert w.isOpened()
    for _ in range(4):w.write(np.zeros((220,240,3),dtype=np.uint8))
    w.release()
    res,_=mod.analizar_clip(clip,intervalo_s=.2)
    assert res['resultados'][0]['continuidad_visual']['evidencia_actual'] is True
    assert res['resultados'][1]['continuidad_visual']['evidencia_actual'] is False
    assert res['resultados'][1]['nivel_revision_visual']=='revision'
    assert res['resumen_visual']['fotogramas_alerta_seguimiento'] >= 1
