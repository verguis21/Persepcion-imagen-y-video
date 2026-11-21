"""Auxiliares de la hoja de etiquetas (sin YOLO)."""
import csv

from scripts.imagen.datos.preparar_etiquetado import COLUMNAS, emparejar_por_iou, iou, leer_etiquetas


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_leer_etiquetas_y_emparejar(tmp_path):
    ruta = tmp_path / 'etiquetas.csv'
    filas = [
        {'clip_id': 'v_s01', 'video_origen': 'v', 'fotograma': 'frame_000.jpg', 'segundo_s': '0.0',
         'tipo_fila': 'fotograma', 'personas_reales': '2', 'celulares_reales': ''},
        {'clip_id': 'v_s01', 'video_origen': 'v', 'fotograma': 'frame_000.jpg', 'segundo_s': '0.0',
         'tipo_fila': 'persona', 'detection_id': '0', 'origen': 'yolo', 'bbox': '[10, 10, 50, 100]',
         'postura': 'erguida'},
        {'clip_id': 'v_s01', 'video_origen': '', 'fotograma': 'frame_000.jpg', 'segundo_s': '0.0',
         'tipo_fila': 'persona', 'origen': 'solo_pose', 'bbox': '[200, 10, 240, 100]'},
    ]
    with open(ruta, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    leidas = leer_etiquetas(ruta)
    assert leidas[0]['personas_reales'] == 2 and leidas[0]['celulares_reales'] is None
    assert leidas[1]['bbox'] == [10.0, 10.0, 50.0, 100.0] and leidas[1]['detection_id'] == 0
    assert leidas[2]['detection_id'] is None and leidas[2]['postura'] is None
    assert leidas[2]['video_origen'] == 'v_s01'          # sin video_origen -> el clip
    assert emparejar_por_iou([12, 12, 50, 100], leidas) is leidas[1]
    assert emparejar_por_iou([300, 300, 320, 320], leidas) is None
