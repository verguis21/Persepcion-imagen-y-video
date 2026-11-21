"""Clasificador de postura con recortes sinteticos (rectangulo vertical vs horizontal)."""
import numpy as np

from app.modulo_imagen.postura.clasificador_postura import cargar_modelo, entrenar, predecir_postura
from app.modulo_imagen.postura.descriptores import vector_hog_lbp


def _recorte(vertical, rng):
    r = np.zeros((128, 64), dtype=np.float32)
    if vertical:
        r[10:118, 24:40] = 255
    else:
        r[56:72, 4:60] = 255
    return np.clip(r + rng.normal(0, 12, r.shape), 0, 255).astype(np.uint8)


def _datos():
    rng = np.random.default_rng(0)
    X, y, grupos = [], [], []
    for i in range(40):
        vertical = i % 2 == 0
        X.append(vector_hog_lbp(_recorte(vertical, rng)))
        y.append('erguida' if vertical else 'horizontal')
        grupos.append(f'video_{i % 8}')
    return np.array(X), np.array(y), np.array(grupos)


def test_entrena_y_separa_las_dos_formas():
    X, y, grupos = _datos()
    modelo, registro = entrenar(X, y, grupos)
    assert (modelo.predict(X) == y).mean() > 0.9
    assert registro['grupos'] == 8 and registro['filas'] == 40
    assert set(registro['mejores_parametros']) == {'svc__C', 'svc__gamma'}


def test_un_solo_grupo_entrena_sin_busqueda():
    X, y, _ = _datos()
    modelo, registro = entrenar(X, y, np.array(['unico'] * len(y)))
    assert registro['f1_macro_validacion_cruzada'] is None and 'sin busqueda' in registro['busqueda']
    assert (modelo.predict(X) == y).mean() > 0.9


def test_sin_modelo_y_caja_degenerada(tmp_path):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert cargar_modelo(tmp_path / 'no_existe.joblib') is None
    assert predecir_postura(frame, [10, 10, 60, 150], None) == 'no_disponible'
    X, y, grupos = _datos()
    modelo, _ = entrenar(X, y, grupos)
    assert predecir_postura(frame, [50, 50, 50, 120], modelo) == 'no_evaluable'
    assert predecir_postura(frame, [10, 10, 60, 150], modelo) in ('erguida', 'horizontal')


def test_guardar_y_cargar(tmp_path):
    import joblib
    X, y, grupos = _datos()
    modelo, _ = entrenar(X, y, grupos)
    joblib.dump(modelo, tmp_path / 'm.joblib')
    assert (cargar_modelo(tmp_path / 'm.joblib').predict(X) == y).mean() > 0.9
