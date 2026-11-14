"""Pruebas sin YOLO real de las señales, calibradas como NO confirmatorias."""
from app.modulo_imagen.senalizacion import agregar_senales, resumen_clip

def test_indicio_postural_solo_recomienda_revision():
    r={'personas':[{'estado_postural':'postura_horizontal_posible'}]}
    agregar_senales(r)
    assert r['nivel_revision_visual']=='revision'
    assert 'Posible incidente' in r['senales_visuales'][0]['titulo']
    assert r['marcha']['estado']=='no_evaluada'

def test_perdida_normal_no_se_vende_como_caida():
    r={'revision_incidente':'perdida_visual_no_concluyente'}
    agregar_senales(r)
    assert r['nivel_revision_visual']=='observacion'
    assert 'caída' not in r['senales_visuales'][0]['titulo'].lower()

def test_evidencia_celular_no_atribuye_distraccion():
    r={'celulares':[{'clase':'cell phone'}]};agregar_senales(r)
    assert r['nivel_revision_visual']=='observacion'
    assert 'no demuestra' in r['senales_visuales'][0]['detalle']

def test_preprocesamiento_sin_correccion_no_afirma_calidad_mejorada():
    r={'preprocesamiento':{'antes':{'sombras_fuertes_estimadas':False,'brillo_bajo_estimado':False,'contraste_bajo_estimado':False},'operaciones':[]}}
    agregar_senales(r)
    assert r['nivel_revision_visual']=='informativo'

def test_resumen_cuenta_frames_no_personas_unicas():
    data=[{'personas':[{},{}],'celulares':[{}],'otros_objetos':[{}],
           'preprocesamiento':{'antes':{'brillo_bajo_estimado':True}}},
          {'personas':[{}],'revision_incidente':'perdida_visual_no_concluyente'}]
    for d in data:agregar_senales(d)
    rs=resumen_clip(data)
    assert rs['detecciones_personas_por_fotograma']==3
    assert rs['detecciones_otros_objetos_por_fotograma']==1
    assert rs['fotogramas_calidad_limitada']==1
    assert rs['fotogramas_revision']==0

def test_pose_roi_horizontal_si_permite_revision_visual():
    r={'recuperacion_roi':{'origen':'roi_pose','estado_postural':'postura_horizontal_posible'}}
    agregar_senales(r)
    assert r['nivel_revision_visual']=='revision'
