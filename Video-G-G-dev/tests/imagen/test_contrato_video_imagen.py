"""Pruebas puras del contrato; no requieren GPU, YOLO ni Flask."""
import json
import pytest
from app.modulo_imagen.entrada.contrato_video import (
    VERSION, leer_metadatos, normalizar_metadatos, contexto_desde_video,
)


def _m(**kwargs):
    return dict(frame_id='f-1', camara_id='cam-A', tiempo_s=1.2, **kwargs)


def test_solo_identificacion_no_inventa_riesgo():
    x = normalizar_metadatos(_m(), ancho=640, alto=480)
    assert x['frame_id'] == 'f-1'
    assert x['contexto_video']['riesgo_reportado_por_video'] is None
    assert x['contexto_video']['tracks'] == []
    assert VERSION == 'imagen-video-1.0.0'


def test_contexto_video_real_de_formato_detect():
    respuesta = {
        'size': [640, 480], 'riesgo': 'MEDIO',
        'detections': [
            {'class_id':3, 'class_name':'Escalera', 'bbox':[10,20,500,400], 'confidence':.89},
            {'class_name':'Persona', 'bbox':[50,50,95,180], 'track_id':7},
            {'tipo':'distraccion', 'risk':'MEDIO'},
        ],
    }
    m = contexto_desde_video(respuesta, frame_id='v:42', tiempo_s=1.4)
    assert m['contexto_video']['escaleras'] == [{'bbox_xyxy_px':[10.,20.,500.,400.], 'confianza':.89}]
    assert m['contexto_video']['tracks'] == [{'track_id':7,'bbox_xyxy_px':[50.,50.,95.,180.]}]
    assert m['contexto_video']['riesgo_reportado_por_video'] == 'MEDIO'
    assert m['frame_id'] == 'v:42'


def test_no_se_inventa_track_si_no_hay():
    x = contexto_desde_video({'size':[100,100],'riesgo':'BAJO','detections':[{'class_name':'Persona','bbox':[0,0,80,90]}]}, frame_id='x')
    assert x['contexto_video']['tracks'] == []


@pytest.mark.parametrize('datos', [
    {}, {'frame_id':''}, {'frame_id':'f','tiempo_s':-1},
    {'frame_id':'f','tiempo_s':float('nan')},
    {'frame_id':'f','contexto_video':{'escaleras':'caja'}},
    {'frame_id':'f','contexto_video':{'riesgo_reportado_por_video':'CAIDA_CONFIRMADA'}},
    {'frame_id':'f','contexto_video':{'tracks':[{'track_id':True}]}},
    {'frame_id':'f','contexto_video':{'escaleras':[{'bbox_xyxy_px':[0,0,999,300]}]}},
])
def test_rechaza_metadatos_invalidos(datos):
    with pytest.raises(ValueError): normalizar_metadatos(datos, ancho=640, alto=480)


def test_clip_requiere_clip_id():
    with pytest.raises(ValueError): normalizar_metadatos(_m(), tipo='clip')
    c = normalizar_metadatos({'clip_id':'C1','tiempo_s':0}, tipo='clip')
    assert c['clip_id'] == 'C1'


def test_lector_json_y_limite():
    assert leer_metadatos(json.dumps(_m()))['frame_id'] == 'f-1'
    with pytest.raises(ValueError): leer_metadatos('[]')
    with pytest.raises(ValueError): leer_metadatos('{' + 'x'*70000 + '}')


def test_rechaza_caja_invertida_e_infinita():
    for caja in ([100,20,10,50],[0,0,float('inf'),50]):
        with pytest.raises(ValueError):
            normalizar_metadatos(_m(contexto_video={'escaleras':[{'bbox_xyxy_px':caja}]}),ancho=640,alto=480)
