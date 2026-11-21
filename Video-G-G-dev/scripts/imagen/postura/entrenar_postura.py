"""Entrena el clasificador de postura con la hoja de etiquetas (etapa 6 del plan).

Uso, desde la carpeta Video-G-G-dev:
    python -m scripts.imagen.postura.entrenar_postura --etiquetas datos_propios/etiquetas.csv

Separa primero los videos de PRUEBA (datos_propios/clips_prueba.txt) y no los
toca: son el examen de la etapa 7. La division es por video original, nunca
por pedazo ni por fotograma.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def elegir_prueba(videos, ruta_lista, escalera_prueba, semilla):
    """Lista de videos originales de prueba; la crea la primera vez."""
    if ruta_lista.is_file():
        return [l.strip() for l in ruta_lista.read_text(encoding='utf-8').splitlines() if l.strip()]
    if escalera_prueba:
        prueba = [v for v in videos if v.startswith(escalera_prueba)]
    else:
        orden = sorted(videos)
        random.Random(semilla).shuffle(orden)
        prueba = orden[:max(1, round(0.2 * len(orden)))]
    if len(prueba) >= len(videos):
        sys.exit('La division dejaria todos los videos en prueba: hacen falta mas videos originales.')
    ruta_lista.parent.mkdir(parents=True, exist_ok=True)
    ruta_lista.write_text('\n'.join(sorted(prueba)) + '\n', encoding='utf-8')
    return sorted(prueba)


def main():
    parser = argparse.ArgumentParser(description='Entrena el SVM de postura (HOG + LBP).')
    parser.add_argument('--etiquetas', required=True)
    parser.add_argument('--fotogramas', default='datos_propios/fotogramas')
    parser.add_argument('--clips-prueba', default='datos_propios/clips_prueba.txt',
                        help='Lista de videos originales reservados para la evaluacion')
    parser.add_argument('--escalera-prueba', help='Prefijo de video_origen a reservar entero como prueba')
    parser.add_argument('--variante', choices=('original', 'restaurada'), default='original')
    parser.add_argument('--salida', default='runs/imagen/postura_svm.joblib')
    parser.add_argument('--semilla', type=int, default=42)
    args = parser.parse_args()

    import cv2
    import joblib
    import numpy as np
    from app.modulo_imagen.postura.clasificador_postura import CLASES, entrenar
    from app.modulo_imagen.postura.descriptores import recorte_persona, vector_hog_lbp
    from scripts.imagen.datos.preparar_etiquetado import leer_etiquetas

    filas = [f for f in leer_etiquetas(args.etiquetas) if f['tipo_fila'] == 'persona']
    no_se_ve = sum(1 for f in filas if f.get('postura') == 'no_se_ve')
    sin_etiqueta = sum(1 for f in filas if not f.get('postura'))
    utiles = [f for f in filas if f.get('postura') in CLASES and f.get('bbox')]
    if not utiles:
        sys.exit('No hay filas de persona con una postura valida en la hoja.')
    videos = sorted({f['video_origen'] for f in utiles})
    prueba = set(elegir_prueba(videos, Path(args.clips_prueba), args.escalera_prueba, args.semilla))
    entrenamiento = [f for f in utiles if f['video_origen'] not in prueba]
    if not entrenamiento:
        sys.exit('Todos los videos etiquetados estan en la lista de prueba.')

    rng = np.random.default_rng(args.semilla)
    cache, X, y, grupos, omitidas = {}, [], [], [], 0
    for f in entrenamiento:
        ruta = Path(args.fotogramas) / f['clip_id'] / f['fotograma']
        if ruta not in cache:
            img = cv2.imread(str(ruta))
            if img is not None and args.variante == 'restaurada':
                from app.modulo_imagen.calidad.nitidez import bloque_calidad
                from app.modulo_imagen.calidad.preprocesamiento import mejorar_imagen
                from app.modulo_imagen.calidad.restauracion import restaurar
                img = restaurar(mejorar_imagen(img)[0], bloque_calidad(img))[0]
            cache = {ruta: img}          # solo el ultimo: la hoja va ordenada por fotograma
        img = cache[ruta]
        recorte = None if img is None else recorte_persona(img, f['bbox'])
        if recorte is None:
            omitidas += 1
            continue
        # Aumento SOLO en entrenamiento: volteo horizontal y brillo +-15 %.
        # Nunca volteo vertical ni rotaciones grandes: cambiarian la postura.
        variantes = [recorte, cv2.flip(recorte, 1),
                     np.clip(recorte.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)]
        for v in variantes:
            X.append(vector_hog_lbp(v))
            y.append(f['postura'])
            grupos.append(f['video_origen'])
    if not X:
        sys.exit('No se pudo leer ningun fotograma de entrenamiento; revise --fotogramas.')
    modelo, registro = entrenar(np.array(X), np.array(y), np.array(grupos))
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, salida)
    resumen = {
        'variante': args.variante,
        'filas': len(entrenamiento) - omitidas,
        'filas_con_aumento': len(y),
        'clips': len({f['clip_id'] for f in entrenamiento}),
        'videos_entrenamiento': sorted({f['video_origen'] for f in entrenamiento}),
        'videos_prueba': sorted(prueba),
        'por_clase_sin_aumento': dict(Counter(f['postura'] for f in entrenamiento)),
        'filas_no_se_ve_excluidas': no_se_ve,
        'filas_sin_etiqueta': sin_etiqueta,
        'filas_omitidas_sin_imagen': omitidas,
        'registro': registro,
    }
    ruta_registro = salida.with_suffix('.json')
    ruta_registro.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f'Modelo:   {salida.resolve()}')
    print(f'Registro: {ruta_registro.resolve()}')
    faltan = [c for c in CLASES if c not in resumen['por_clase_sin_aumento']]
    if faltan:
        print(f'Aviso: el modelo no vio ningun ejemplo de {", ".join(faltan)}; nunca respondera esas clases.')


if __name__ == '__main__':
    main()
