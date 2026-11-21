"""Clasificador de postura propio: HOG + LBP -> SVM (etapa 6 del plan, sesiones 6 y 11).

Pipeline: escalar -> PCA (95 % de la varianza) -> SVM con kernel RBF y pesos
balanceados (casi todo sera "erguida"; sin balancear, el modelo ignoraria las
demas clases). C y gamma se eligen con validacion cruzada AGRUPADA por video
original, para que el modelo no se evalue con fotos casi iguales a las que vio.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..postura.descriptores import recorte_persona, vector_hog_lbp

CLASES = ('erguida', 'inclinada', 'agachada_o_sentada', 'horizontal')
RUTA_MODELO = Path('runs/imagen/postura_svm.joblib')
REJILLA = {'svc__C': [0.5, 2.0, 8.0], 'svc__gamma': ['scale', 0.001]}


def crear_pipeline(C=1.0, gamma='scale'):
    return Pipeline([('escala', StandardScaler()),
                     ('pca', PCA(n_components=0.95, svd_solver='full')),
                     ('svc', SVC(kernel='rbf', class_weight='balanced', C=C, gamma=gamma))])


def entrenar(X, y, grupos):
    """Ajusta el pipeline. Devuelve (pipeline, registro con lo que se hizo)."""
    X, y, grupos = np.asarray(X), np.asarray(y), np.asarray(grupos)
    n_grupos = len(set(grupos.tolist()))
    registro = {'filas': int(len(y)), 'grupos': n_grupos,
                'por_clase': dict(Counter(y.tolist())), 'busqueda': None}
    if n_grupos < 2 or len(set(y.tolist())) < 2:
        modelo = crear_pipeline().fit(X, y)
        registro['busqueda'] = 'sin busqueda: hace falta al menos 2 videos y 2 clases'
        registro['mejores_parametros'] = {'svc__C': 1.0, 'svc__gamma': 'scale'}
        registro['f1_macro_validacion_cruzada'] = None
        return modelo, registro
    busqueda = GridSearchCV(crear_pipeline(), REJILLA, scoring='f1_macro',
                            cv=GroupKFold(n_splits=min(5, n_grupos)), error_score=np.nan)
    busqueda.fit(X, y, groups=grupos)
    registro['busqueda'] = f'GridSearchCV con GroupKFold({min(5, n_grupos)}) agrupado por video'
    registro['mejores_parametros'] = busqueda.best_params_
    puntuacion = busqueda.best_score_
    registro['f1_macro_validacion_cruzada'] = None if np.isnan(puntuacion) else round(float(puntuacion), 3)
    return busqueda.best_estimator_, registro


def cargar_modelo(ruta=RUTA_MODELO):
    """El modelo entrenado, o None si todavia no existe."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return None
    import joblib
    return joblib.load(ruta)


def variante_del_modelo(ruta=RUTA_MODELO):
    """Variante de imagen con la que se entreno ('original' o 'restaurada').

    Se lee del registro JSON que guarda el script de entrenamiento junto al
    modelo, para usar en produccion la MISMA variante que en el entrenamiento.
    """
    import json
    registro = Path(ruta).with_suffix('.json')
    try:
        return json.loads(registro.read_text(encoding='utf-8')).get('variante') or 'original'
    except (OSError, ValueError):
        return 'original'


def predecir_postura(frame, bbox, modelo):
    """'no_disponible' sin modelo, 'no_evaluable' si la caja no sirve, o la clase."""
    if modelo is None:
        return 'no_disponible'
    recorte = recorte_persona(frame, bbox)
    if recorte is None:
        return 'no_evaluable'
    return str(modelo.predict(vector_hog_lbp(recorte).reshape(1, -1))[0])
