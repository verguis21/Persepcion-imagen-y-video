"""Pruebas del contrato imagen-evidencia-1.0 sin cargar YOLO ni pesos."""
import copy
import json
from pathlib import Path

import pytest

from app.modulo_imagen.evidencia import (
    VERSION,
    construir_evidencia,
    construir_fotograma,
    personas_de_dato,
    validar_evidencia,
)

RUTA_FIXTURE = Path(__file__).parent / 'fixtures' / 'evidencia_ejemplo.json'


def _fixture():
    return json.loads(RUTA_FIXTURE.read_text(encoding='utf-8'))


def _pose(bbox, inclinacion=38.0):
    return {'persona_indice': 0, 'bbox': list(bbox), 'inclinacion_torso': inclinacion,
            'puntos_corporales': [[0.0, 0.0, 0.9] for _ in range(17)]}


def _dato_minimo(**extra):
    """Forma reducida de lo que devuelve analizar_frame (más segundo_s e índice del clip)."""
    dato = {
        'personas': [{
            'detection_id': 0, 'bbox': [210, 140, 330, 520], 'confianza': 0.88,
            'pose': _pose([212, 142, 328, 518]), 'puntos_visibles': {}, 'pose_asociada': True,
            'estado_postural': 'inclinacion_para_revision',
        }],
        'poses_sin_asociar': [],
        'celulares': [{'clase': 'cell phone', 'bbox': [301, 262, 322, 300], 'confianza': 0.47,
                       'relacion_manos': {'estado': 'cerca_de_mano', 'persona_id': 0,
                                          'distancia_relativa_mano': 0.05}}],
        'otros_objetos': [{'clase': 'backpack', 'bbox': [400, 600, 470, 680], 'confianza': 0.61}],
        'calidad_original': {'luminosidad': 96.2, 'contraste': 41.0, 'nitidez_laplaciana': 212.5,
                             'ruido_mad_estimado': 2.0, 'brillo_bajo_estimado': False},
        'segundo_s': 0.0,
        'fotograma_indice': 0,
    }
    dato.update(extra)
    return dato


def test_cabecera_no_importa_modulos_con_yolo():
    import app.modulo_imagen.evidencia as modulo
    cabecera = Path(modulo.__file__).read_text(encoding='utf-8').split('\ndef ')[0]
    for nombre in ('detector', 'pose', 'analisis_tiempo_real', 'analizador_clip'):
        assert f'from .{nombre} import' not in cabecera
        assert f'import {nombre}' not in cabecera


def test_fixture_valida():
    evidencia = _fixture()
    assert evidencia['schema_version'] == VERSION
    validar_evidencia(evidencia)


def test_fixture_con_deteccion_restaurada_nula_valida():
    evidencia = _fixture()
    evidencia['fotogramas'][0]['deteccion_restaurada'] = None
    validar_evidencia(evidencia)


def test_clave_prohibida_dentro_de_persona_lanza():
    evidencia = _fixture()
    evidencia['fotogramas'][0]['personas'][0]['riesgo'] = 'alto'
    with pytest.raises(ValueError):
        validar_evidencia(evidencia)


def test_clave_prohibida_por_subcadena_y_mayusculas_lanza():
    evidencia = _fixture()
    evidencia['fotogramas'][0]['objetos'][0]['Nivel_Alerta'] = 1
    with pytest.raises(ValueError):
        validar_evidencia(evidencia)


def test_valor_cerrado_no_permitido_lanza():
    evidencia = _fixture()
    evidencia['fotogramas'][0]['personas'][0]['contacto_pasamanos'] = 'quizas'
    with pytest.raises(ValueError):
        validar_evidencia(evidencia)


@pytest.mark.parametrize('ruta', [
    ('schema_version',), ('clip', 'clip_id'), ('fotogramas',),
])
def test_falta_campo_obligatorio_lanza(ruta):
    evidencia = _fixture()
    destino = evidencia
    for clave in ruta[:-1]:
        destino = destino[clave]
    del destino[ruta[-1]]
    with pytest.raises(ValueError):
        validar_evidencia(evidencia)


def test_version_distinta_lanza():
    evidencia = _fixture()
    evidencia['schema_version'] = 'imagen-evidencia-0.9'
    with pytest.raises(ValueError):
        validar_evidencia(evidencia)


def test_construir_fotograma_renombra_y_deja_bloques_nulos():
    fotograma = construir_fotograma(_dato_minimo())
    assert fotograma['indice'] == 0
    assert fotograma['segundo_s'] == 0.0
    persona = fotograma['personas'][0]
    assert persona['postura_reglas_pose'] == 'inclinacion_para_revision'
    assert persona['confianza_detector'] == 0.88
    assert 'estado_postural' not in persona and 'confianza' not in persona
    assert persona['origen'] == 'yolo'
    assert persona['detection_id'] == 0
    assert persona['inclinacion_torso_grados'] == 38.0
    assert persona['postura_clasificador'] == 'no_disponible'
    assert persona['concordancia_posturas'] is None
    assert persona['en_zona_escalones'] is None
    assert persona['contacto_pasamanos'] == 'no_evaluable'
    assert persona['objetos_en_mano'] == []
    assert fotograma['restauracion'] is None
    assert fotograma['escena'] is None
    assert fotograma['deteccion_restaurada'] is None
    assert fotograma['calidad']['nitidez_laplaciana'] == 212.5
    assert fotograma['calidad']['apto_para_detalle'] is None
    assert 'brillo_bajo_estimado' not in fotograma['calidad']
    clases = [o['clase'] for o in fotograma['objetos']]
    assert clases == ['cell phone', 'backpack']
    assert all('relacion_manos' not in o for o in fotograma['objetos'])
    assert fotograma['objetos'][0]['confianza_detector'] == 0.47


def test_poses_sin_asociar_entran_como_solo_pose():
    pose = {**_pose([50, 300, 250, 380], inclinacion=70.0), 'origen': 'solo_pose',
            'estado_postural': 'postura_horizontal_posible'}
    dato = _dato_minimo(poses_sin_asociar=[pose])
    personas = construir_fotograma(dato)['personas']
    assert len(personas) == 2
    extra = personas[1]
    assert extra['origen'] == 'solo_pose'
    assert extra['detection_id'] is None
    assert extra['confianza_detector'] is None
    assert extra['bbox'] == [50, 300, 250, 380]
    assert extra['postura_reglas_pose'] == 'postura_horizontal_posible'
    assert extra['inclinacion_torso_grados'] == 70.0


def test_recuperacion_roi_yolo_entra_con_su_origen():
    roi = {'bbox': [10, 20, 90, 60], 'confianza': 0.31, 'origen': 'roi_yolo', 'clase': 'person',
           'roi_usada': [0, 0, 240, 240], 'tiempo_roi_ms': 3.0}
    dato = _dato_minimo(personas=[], recuperacion_roi=roi)
    personas = construir_fotograma(dato)['personas']
    assert len(personas) == 1
    assert personas[0]['origen'] == 'roi_yolo'
    assert personas[0]['detection_id'] is None
    assert personas[0]['confianza_detector'] == 0.31
    assert personas[0]['postura_reglas_pose'] == 'no_evaluable'
    assert personas[0]['inclinacion_torso_grados'] is None


def test_recuperacion_roi_pose_y_sin_origen():
    pose = _pose([10, 20, 90, 60], inclinacion=60.0)
    roi = {'bbox': [10, 20, 90, 60], 'origen': 'roi_pose', 'clase': 'person', 'pose': pose,
           'estado_postural': 'postura_horizontal_posible', 'confianza': None}
    personas = personas_de_dato(_dato_minimo(recuperacion_roi=roi))
    assert [p['origen'] for p in personas] == ['yolo', 'roi_pose']
    assert personas[1]['pose'] is pose and personas[1]['confianza'] is None
    sin = personas_de_dato(_dato_minimo(recuperacion_roi={'estado': 'sin_deteccion_actual'}))
    assert [p['origen'] for p in sin] == ['yolo']


def test_orden_fijo_de_personas_de_dato():
    huerfana = {**_pose([0, 0, 10, 10]), 'origen': 'solo_pose', 'estado_postural': 'sin_indicio_suficiente'}
    roi = {'bbox': [1, 1, 5, 5], 'confianza': 0.2, 'origen': 'roi_yolo', 'clase': 'person'}
    dato = _dato_minimo(poses_sin_asociar=[huerfana], recuperacion_roi=roi)
    assert [p['origen'] for p in personas_de_dato(dato)] == ['yolo', 'solo_pose', 'roi_yolo']


def test_concordancia_con_clasificador():
    dato = _dato_minimo(evidencia_imagen={'calidad': None, 'restauracion': None,
                                          'deteccion_restaurada': None, 'escena': None,
                                          'calibracion': None,
                                          'posturas_clasificador': ['inclinada']})
    persona = construir_fotograma(dato)['personas'][0]
    assert persona['postura_clasificador'] == 'inclinada'
    assert persona['concordancia_posturas'] is True
    dato['evidencia_imagen']['posturas_clasificador'] = ['erguida']
    assert construir_fotograma(dato)['personas'][0]['concordancia_posturas'] is False
    dato['evidencia_imagen']['posturas_clasificador'] = ['agachada_o_sentada']
    assert construir_fotograma(dato)['personas'][0]['concordancia_posturas'] is None


def test_bloques_de_evidencia_imagen_se_usan_si_existen():
    calidad = {'luminosidad': 1.0, 'contraste': 2.0, 'nitidez_laplaciana': 3.0,
               'energia_alta_frecuencia': 0.1, 'ruido_mad_estimado': 0.5,
               'desenfoque_movimiento': {'anisotropia': 0.2, 'direccion_grados': 0.0},
               'apto_para_detalle': False}
    escena = {'origen_zona': 'no_disponible', 'zona_escalones': None,
              'lineas_escalon_detectadas': 0, 'pasamanos': None}
    dato = _dato_minimo(evidencia_imagen={'calidad': calidad, 'restauracion': {'operaciones': []},
                                          'deteccion_restaurada': None, 'escena': escena,
                                          'calibracion': None, 'posturas_clasificador': None})
    fotograma = construir_fotograma(dato)
    assert fotograma['calidad'] == calidad
    assert fotograma['escena'] == escena
    assert fotograma['restauracion'] == {'operaciones': []}
    assert fotograma['personas'][0]['postura_clasificador'] == 'no_disponible'


def _resultado_clip(resultados, contexto=None):
    return {'modulo': 'imagen', 'estado': 'ok', 'clip': 'x.mp4', 'duracion_s': 3.0,
            'fps_origen': 30.0, 'fotogramas_planificados': len(resultados),
            'contexto_video': contexto if contexto is not None else {},
            'resumen_visual': {'fotogramas_alerta_seguimiento': 0},
            'resultados': resultados,
            'nota': 'No confirma caídas.'}


def test_construir_evidencia_no_copia_riesgo_de_video_y_valida():
    contexto = {'escaleras': [{'bbox_xyxy_px': [0, 0, 100, 100], 'confianza': 0.9}],
                'tracks': [{'track_id': 3}], 'riesgo_reportado_por_video': 'ALTO'}
    evidencia = construir_evidencia(_resultado_clip([_dato_minimo()], contexto),
                                    clip_id='c1', camara_id='escalera_g')
    assert evidencia['schema_version'] == VERSION
    assert evidencia['clip']['clip_id'] == 'c1'
    assert evidencia['clip']['camara_id'] == 'escalera_g'
    assert evidencia['clip']['duracion_s'] == 3.0
    assert evidencia['clip']['fps_origen'] == 30.0
    assert set(evidencia['clip']['contexto_video']) == {'escaleras', 'tracks'}
    assert evidencia['clip']['contexto_video']['tracks'] == [{'track_id': 3}]
    assert 'riesgo_reportado_por_video' not in json.dumps(evidencia)
    for clave in ('resumen_visual', 'senales_visuales', 'nota'):
        assert clave not in evidencia
    assert evidencia['limitaciones']
    validar_evidencia(evidencia)


def test_construir_evidencia_sin_contexto_usa_listas_vacias():
    resultado = _resultado_clip([_dato_minimo()])
    del resultado['contexto_video']
    evidencia = construir_evidencia(resultado, clip_id='c2', camara_id='cam')
    assert evidencia['clip']['contexto_video'] == {'escaleras': [], 'tracks': []}
    validar_evidencia(evidencia)


def test_fotograma_no_decodificado_se_omite_y_resumen():
    segundo = _dato_minimo(segundo_s=1.0, fotograma_indice=30, evidencia_imagen={
        'calidad': {'luminosidad': 1.0, 'contraste': 1.0, 'nitidez_laplaciana': 1.0,
                    'energia_alta_frecuencia': 0.0, 'ruido_mad_estimado': 0.0,
                    'desenfoque_movimiento': None, 'apto_para_detalle': False},
        'restauracion': None, 'deteccion_restaurada': None, 'escena': None,
        'calibracion': None, 'posturas_clasificador': ['inclinada']})
    resultados = [_dato_minimo(), segundo, {'segundo_s': 2.0, 'estado': 'no_decodificado'}]
    evidencia = construir_evidencia(_resultado_clip(copy.deepcopy(resultados)),
                                    clip_id='c3', camara_id='cam')
    assert [f['segundo_s'] for f in evidencia['fotogramas']] == [0.0, 1.0]
    resumen = evidencia['resumen']
    assert resumen['fotogramas_analizados'] == 2
    assert resumen['personas_max'] == 1
    assert resumen['fotogramas_con_desenfoque'] == 1
    assert resumen['posturas_observadas'] == {'inclinada': 1}
    assert resumen['clases_objeto'] == {'cell phone': 2, 'backpack': 2}
    validar_evidencia(evidencia)
