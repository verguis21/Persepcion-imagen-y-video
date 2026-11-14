"""Prueba del uso del respaldo Pose cuando falla ROI-YOLO, sin YOLO real."""
import sys
from types import SimpleNamespace
import importlib
import cv2
import numpy as np


def test_rescate_pose_persiste_como_evidencia_no_confirmada(tmp_path, monkeypatch):
    # Emula el módulo de análisis para validar el flujo sin requerir ultralytics.
    modulo='app.modulo_imagen.analisis_tiempo_real'
    monkeypatch.setitem(sys.modules, modulo, SimpleNamespace(analizar_frame=lambda frame,modo=None:(
        {'personas':[], 'poses_sin_asociar':[], 'advertencias':[], 'tiempos_ms':{}},frame.copy())))
    import app.modulo_imagen.analizador_clip as m
    # Alterna un primer fotograma que genera observación, seguido de pérdidas.
    contador={'i':0}
    def fake(frame, modo=None):
        contador['i']+=1
        return ({'personas':[{'bbox':[100,60,175,155], 'estado_postural':'postura_horizontal_posible'}]
                            if contador['i']==1 else [],
                 'poses_sin_asociar':[], 'advertencias':[], 'tiempos_ms':{}},frame.copy())
    monkeypatch.setattr(m,'analizar_frame',fake)
    monkeypatch.setattr(m,'recuperar_en_roi',lambda frame,bbox:None)
    monkeypatch.setattr(m,'recuperar_pose_en_roi',lambda frame,bbox:{'origen':'roi_pose','bbox':[101,62,176,157],'confianza':None})
    clip=tmp_path/'prueba.avi'
    writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),5.0,(320,240))
    assert writer.isOpened()
    for _ in range(5):writer.write(np.zeros((240,320,3),dtype=np.uint8))
    writer.release()
    salida,_=m.analizar_clip(clip,intervalo_s=0.2)
    assert salida['fotogramas_analizados']==5
    assert salida['resultados'][0]['revision_incidente']=='indicio_visual_sin_confirmar'
    segundo=salida['resultados'][1]
    assert segundo['recuperacion_roi']['origen']=='roi_pose'
    assert segundo['revision_incidente']=='persona_recuperada_pose_roi_sin_confirmar_incidente'
    assert 'roi_yolo' in segundo['tiempos_ms'] and 'roi_pose' in segundo['tiempos_ms']
    assert 'total_frame_con_roi' in segundo['tiempos_ms']
