"""Pruebas de relaciones persona-objeto-pasamanos con poses sinteticas."""
from app.modulo_imagen.relaciones import contacto_pasamanos, objetos_en_mano

SEGMENTO = [200, 0, 200, 400]          # pasamanos vertical en x = 200


def persona(izquierda=None, derecha=None, confianza=0.9):
    """Persona de 300 px de alto con 17 puntos COCO; solo importan las munecas."""
    puntos = [[0.0, 0.0, 0.0] for _ in range(17)]
    if izquierda:
        puntos[9] = [izquierda[0], izquierda[1], confianza]
    if derecha:
        puntos[10] = [derecha[0], derecha[1], confianza]
    return {'bbox': [50, 50, 150, 350], 'pose': {'puntos_corporales': puntos}}


def test_muneca_junto_al_pasamanos():
    assert contacto_pasamanos(persona(derecha=(195, 200)), SEGMENTO) == 'si'


def test_muneca_lejos_del_pasamanos():
    assert contacto_pasamanos(persona(derecha=(50, 200)), SEGMENTO) == 'no'


def test_sin_munecas_o_sin_segmento_no_es_evaluable():
    assert contacto_pasamanos(persona(derecha=(195, 200), confianza=0.1), SEGMENTO) == 'no_evaluable'
    assert contacto_pasamanos(persona(derecha=(195, 200)), None) == 'no_evaluable'
    assert contacto_pasamanos({'bbox': [50, 50, 150, 350], 'pose': None}, SEGMENTO) == 'no_evaluable'


def test_varios_pasamanos_basta_estar_cerca_de_uno():
    dos = [[200, 0, 200, 400], [20, 0, 20, 400]]
    assert contacto_pasamanos(persona(izquierda=(25, 200)), dos) == 'si'      # cerca del segundo
    assert contacto_pasamanos(persona(izquierda=(110, 200)), dos) == 'no'     # lejos de ambos
    assert contacto_pasamanos(persona(izquierda=(25, 200)), []) == 'no_evaluable'


def test_celular_en_la_mano_derecha():
    p = persona(izquierda=(60, 200), derecha=(140, 200))
    objetos = [{'clase': 'cell phone', 'bbox': [130, 190, 150, 210], 'confianza': 0.47}]
    resultado = objetos_en_mano(p, objetos)
    assert len(resultado) == 1
    assert resultado[0]['mano'] == 'derecha' and resultado[0]['indice_objeto'] == 0
    assert resultado[0]['confianza_detector'] == 0.47


def test_objeto_lejano_persona_o_sin_pose():
    p = persona(derecha=(140, 200))
    assert objetos_en_mano(p, [{'clase': 'backpack', 'bbox': [600, 600, 660, 680], 'confianza': 0.6}]) == []
    assert objetos_en_mano(p, [{'clase': 'person', 'bbox': [130, 190, 150, 210], 'confianza': 0.9}]) == []
    assert objetos_en_mano({'bbox': (50, 50, 150, 350), 'pose': None},
                           [{'clase': 'cell phone', 'bbox': [130, 190, 150, 210], 'confianza': 0.5}]) == []
