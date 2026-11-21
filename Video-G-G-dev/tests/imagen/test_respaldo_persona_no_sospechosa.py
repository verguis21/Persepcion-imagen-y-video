import importlib
import sys
from types import SimpleNamespace

import cv2
import numpy as np

def test_recuperacion_pose_tras_desaparicion_aun_sin_postura_previa(monkeypatch,tmp_path):
    # Se prueba el flujo de integración, NO la calidad ni precisión de los pesos YOLO.
    monkeypatch.setitem(sys.modules, 'app.modulo_imagen.deteccion.analisis_tiempo_real',
                        SimpleNamespace(analizar_frame=lambda *a,**k:None))
    m=importlib.import_module('app.modulo_imagen.entrada.analizador_clip')
    cuenta={'n':0}
    def fake(frame, modo=None):
        cuenta['n']+=1
        visible=[{'bbox':[100,75,142,184], 'estado_postural':'sin_indicio_suficiente'}] if cuenta['n']==1 else []
        return {'personas':visible, 'poses_sin_asociar':[], 'advertencias':[],
                'tiempos_ms':{}, 'estado':'ok'}, frame.copy()
    roi_calls=[]
    monkeypatch.setattr(m,'analizar_frame',fake)
    monkeypatch.setattr(m,'recuperar_en_roi',lambda frame,bbox:roi_calls.append(list(bbox)) or None)
    monkeypatch.setattr(m,'recuperar_pose_en_roi',lambda frame,bbox:{'origen':'roi_pose',
        'bbox':[96,145,197,196], 'estado_postural':'postura_horizontal_posible','confianza':None})
    output=tmp_path/'evidencia.avi'; writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'MJPG'),5.,(320,240))
    assert writer.isOpened()
    for _ in range(3): writer.write(np.zeros((240,320,3),dtype=np.uint8))
    writer.release()
    result,_=m.analizar_clip(output,intervalo_s=0.2)
    assert result['fotogramas_analizados']==3
    assert len(roi_calls)>=1
    assert result['resultados'][0]['nivel_revision_visual']=='informativo'
    assert result['resultados'][1]['recuperacion_roi']['origen']=='roi_pose'
    assert result['resultados'][1]['nivel_revision_visual']=='revision'
    assert result['resumen_visual']['fotogramas_revision']>=1

def test_candidato_en_borde_se_descarta():
    m=importlib.import_module('app.modulo_imagen.entrada.analizador_clip')
    prev=[{'bbox':[250,10,320,238],'segundo_s':0.}]
    assert m._posible_roi_por_desaparicion(prev,[],(240,320,3),.2) is None
