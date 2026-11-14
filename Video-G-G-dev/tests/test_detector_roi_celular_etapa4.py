"""Pruebas aisladas con salida ficticia de YOLO, NO prueba de precisión real."""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np


def cargar_modulo(monkeypatch, caja_local):
    class ModeloFalso:
        names={0:'person',67:'cell phone'}
        def __init__(self):
            self.llamadas=0
        def predict(self, **kwargs):
            self.llamadas+=1
            assert kwargs['classes']==[67]
            assert kwargs['conf']==0.23
            return [SimpleNamespace(boxes=caja_local)]
    modelo=ModeloFalso()
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda _:modelo))
    fichero=pathlib.Path(__file__).resolve().parents[1]/'app/modulo_imagen/detector.py'
    spec=importlib.util.spec_from_file_location('detector_test_aislado',fichero)
    modulo=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo,modelo


def persona_con_pose(muneca_conf):
    puntos=[[0,0,0] for _ in range(17)]
    puntos[9]=[100,100,muneca_conf]
    return [{'bbox':[65,40,140,220],'pose':{'puntos_corporales':puntos}}]


def test_busqueda_escala_celular_y_mapea_caja(monkeypatch):
    caja=SimpleNamespace(xyxy=[np.array([40.,45.,55.,60.])],conf=[np.float32(.31)])
    mod,modelo=cargar_modulo(monkeypatch,[caja])
    frame=np.zeros((250,300,3),dtype=np.uint8)
    res=mod.buscar_celulares_en_manos(frame,persona_con_pose(.8))
    assert len(res)==1
    assert res[0]['clase']=='cell phone'
    assert res[0]['origen']=='roi_mano_exploratorio'
    assert modelo.llamadas==1
    assert res[0]['bbox'][0]>=0 and res[0]['bbox'][2]<=300


def test_sin_munecas_confiables_no_alucina_ni_ejecuta_yolo(monkeypatch):
    mod,modelo=cargar_modulo(monkeypatch,[])
    frame=np.zeros((250,300,3),dtype=np.uint8)
    assert mod.buscar_celulares_en_manos(frame,persona_con_pose(.2))==[]
    assert modelo.llamadas==0
