"""Pruebas de geometría ROI-Pose sin exigir carga de modelos YOLO."""
import sys
from types import SimpleNamespace
import numpy as np
from app.modulo_imagen.deteccion.pose_roi import recuperar_pose_en_roi


def _pos(x1=60, y1=50):
    return {'bbox':[x1,y1,x1+90,y1+50],
            'inclinacion_torso':70,
            'puntos_corporales':[[x1+3+i,y1+7+i,0.9] for i in range(17)]}


def test_pose_roi_convierte_caja_y_articulaciones_a_global(monkeypatch):
    monkeypatch.setitem(sys.modules,'app.modulo_imagen.deteccion.analisis_tiempo_real',
                        SimpleNamespace(_evaluar_postura=lambda p:'postura_horizontal_posible'))
    frame=np.zeros((480,848,3),dtype=np.uint8)
    bbox=[250,220,350,300]
    capturado={}
    def dummy(crop):
        capturado['shape']=crop.shape
        return crop,[_pos()]
    r=recuperar_pose_en_roi(frame,bbox,analizador_pose=dummy)
    assert r is not None
    assert r['origen']=='roi_pose'
    assert r['confianza'] is None
    x,y,_,_=r['roi_usada']
    assert r['bbox']==[60+x,50+y,150+x,100+y]
    assert r['pose']['puntos_corporales'][0][:2]==[63+x,57+y]
    assert r['puntos_visibles_cantidad']==17
    assert r['estado_postural']=='postura_horizontal_posible'
    assert capturado['shape'][0]>0


def test_pose_roi_rechaza_puntos_invisibles():
    frame=np.zeros((480,848,3),dtype=np.uint8)
    p=_pos(); p['puntos_corporales']=[[60,50,0.1] for _ in range(17)]
    assert recuperar_pose_en_roi(frame,[250,220,350,300],analizador_pose=lambda im:(im,[p])) is None


def test_pose_roi_rechaza_deteccion_distante():
    frame=np.zeros((480,848,3),dtype=np.uint8)
    assert recuperar_pose_en_roi(frame,[250,220,350,300],analizador_pose=lambda im:(im,[_pos(0,0)])) is None


def test_pose_roi_no_inventa_caja_si_no_detecta():
    frame=np.zeros((480,848,3),dtype=np.uint8)
    assert recuperar_pose_en_roi(frame,[250,220,350,300],analizador_pose=lambda im:(im,[])) is None


def test_pose_roi_rechaza_bbox_invalida():
    frame=np.zeros((480,848,3),dtype=np.uint8)
    assert recuperar_pose_en_roi(frame,[250,220],analizador_pose=lambda im:(im,[])) is None
