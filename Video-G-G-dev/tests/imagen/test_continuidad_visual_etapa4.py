"""Pruebas con datos fabricados del contrato, NO precisión de YOLO."""
from app.modulo_imagen.deteccion.continuidad_visual import ContinuidadVisual
from app.modulo_imagen.salida.senalizacion import agregar_senales, resumen_clip


def dato(t, postura=None):
    return {'segundo_s':t,'personas':([{'bbox':[30,40,120,190], 'estado_postural':postura}]
                                       if postura else []),
            'poses_sin_asociar':[],'advertencias':[], 'tiempos_ms':{}}


def test_alarma_no_aparece_sin_evidencia_previa():
    c=ContinuidadVisual()
    r=c.aplicar(dato(0.2))
    agregar_senales(r)
    assert r['nivel_revision_visual']=='informativo'
    assert 'continuidad_visual' not in r


def test_reconoce_evidencia_actual_y_la_sostiene_como_historial():
    c=ContinuidadVisual(persistencia_s=2)
    a=c.aplicar(dato(1.0,'postura_horizontal_posible'))
    agregar_senales(a)
    assert a['nivel_revision_visual']=='revision'
    assert a['continuidad_visual']['evidencia_actual'] is True
    b=c.aplicar(dato(1.2))
    agregar_senales(b)
    assert b['nivel_revision_visual']=='revision'
    assert b['continuidad_visual']['evidencia_actual'] is False
    assert any(s['codigo']=='seguimiento_indicio_previo' for s in b['senales_visuales'])
    assert resumen_clip([a,b])['fotogramas_alerta_seguimiento']==1


def test_caduca_y_no_se_inventa_presencia_en_suelo():
    c=ContinuidadVisual(persistencia_s=2)
    c.aplicar(dato(0.0,'postura_horizontal_posible'))
    r=c.aplicar(dato(2.5))
    agregar_senales(r)
    assert 'continuidad_visual' not in r
    assert r['nivel_revision_visual']=='informativo'


def test_postura_horizontal_continua_renueva_ventana():
    c=ContinuidadVisual(2)
    c.aplicar(dato(0,'postura_horizontal_posible'))
    c.aplicar(dato(1.5,'postura_horizontal_posible'))
    r=c.aplicar(dato(3.0))
    assert r['continuidad_visual']['segundos_desde_evidencia']==1.5


def test_no_alerta_falsa_por_solo_celular_sin_mano():
    r={'celulares':[{'clase':'cell phone', 'relacion_manos':{'estado':'no_evaluable'}}]}
    agregar_senales(r)
    assert r['nivel_revision_visual']=='observacion'


def test_celular_cercano_mano_marcado_para_revision_no_confirmacion():
    r={'celulares':[{'clase':'cell phone','relacion_manos':{'estado':'cerca_de_mano'}}]}
    agregar_senales(r)
    assert r['nivel_revision_visual']=='revision'
    assert 'no demuestra' in r['senales_visuales'][0]['detalle']
