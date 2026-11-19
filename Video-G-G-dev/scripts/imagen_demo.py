"""Demo del modulo Imagen sobre un clip: evidencia.json y una lamina de tres vistas.

Uso, desde la carpeta Video-G-G-dev:
    python -m scripts.imagen_demo clip.mp4 --camara escalera_g

Por cada fotograma analizado la lamina muestra: original, restaurado y anotado
(cajas y esqueleto, mas la zona de escalones y el pasamanos si hay calibracion).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

ANCHO_VISTA = 520


def _vista(imagen, titulo):
    import cv2
    escala = ANCHO_VISTA / imagen.shape[1]
    vista = cv2.resize(imagen, (ANCHO_VISTA, int(round(imagen.shape[0] * escala))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(vista, (0, 0), (ANCHO_VISTA, 24), (30, 30, 30), -1)
    cv2.putText(vista, titulo, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vista


def main():
    parser = argparse.ArgumentParser(description='Evidencia y lamina de demostracion para un clip.')
    parser.add_argument('clip', help='Clip de hasta 10 segundos')
    parser.add_argument('--camara', default='camara_local', help='camara_id pactado con Video (para la calibracion)')
    parser.add_argument('--modelo', default='runs/imagen/postura_svm.joblib')
    parser.add_argument('--intervalo', type=float, default=1.0)
    parser.add_argument('--salida', default='runs/imagen/demo')
    parser.add_argument('--deteccion-doble', action='store_true', help='Detectar tambien sobre el fotograma restaurado')
    args = parser.parse_args()

    import cv2
    import numpy as np
    from app.modulo_imagen.analizador_clip import analizar_clip
    from app.modulo_imagen.escena import cargar_calibracion
    from app.modulo_imagen.evidencia import (DETECCION_DOBLE, construir_evidencia, crear_enriquecedor,
                                             validar_evidencia)
    from app.modulo_imagen.nitidez import bloque_calidad
    from app.modulo_imagen.preprocesamiento import mejorar_imagen
    from app.modulo_imagen.restauracion import restaurar

    modelo = None
    try:
        from app.modulo_imagen.clasificador_postura import cargar_modelo
        modelo = cargar_modelo(Path(args.modelo))
    except ImportError:
        print('Clasificador de postura aun no programado: se usa solo la regla de pose.')
    calibracion = cargar_calibracion(args.camara)
    if calibracion is None:
        print(f"Sin calibracion para la camara '{args.camara}': escalones y pasamanos saldran vacios.")

    clip = Path(args.clip)
    carpeta = Path(args.salida) / clip.stem
    carpeta.mkdir(parents=True, exist_ok=True)
    enriquecer = crear_enriquecedor(modelo_postura=modelo, calibracion=calibracion,
                                    deteccion_doble=args.deteccion_doble or DETECCION_DOBLE)
    with tempfile.TemporaryDirectory(prefix='demo_imagen_') as temporal:
        resultado, _ = analizar_clip(clip, intervalo_s=args.intervalo, guardar_en=temporal, enriquecer=enriquecer)
        evidencia = construir_evidencia(resultado, clip_id=clip.stem, camara_id=args.camara)
        validar_evidencia(evidencia)
        filas = []
        for i, fotograma in enumerate(evidencia['fotogramas']):
            original = cv2.imread(str(Path(temporal) / f'frame_{i:03d}_original.jpg'))
            anotada = cv2.imread(str(Path(temporal) / f'frame_{i:03d}_anotado.jpg'))
            if original is None or anotada is None:
                continue
            restaurada = restaurar(mejorar_imagen(original)[0], bloque_calidad(original))[0]
            escena = fotograma.get('escena') or {}
            if escena.get('zona_escalones'):
                cv2.polylines(anotada, [np.array(escena['zona_escalones'], dtype=np.int32)], True, (60, 200, 60), 2)
            for segmento in (escena.get('pasamanos') or {}).get('segmentos') or []:
                cv2.line(anotada, tuple(segmento[:2]), tuple(segmento[2:]), (40, 140, 255), 3)
            operaciones = ', '.join((fotograma.get('restauracion') or {}).get('operaciones') or []) or 'sin cambios'
            filas.append(np.hstack([_vista(original, f"original  t={fotograma['segundo_s']} s"),
                                    _vista(restaurada, f'restaurado ({operaciones})'),
                                    _vista(anotada, 'anotado')]))
    (carpeta / 'evidencia.json').write_text(json.dumps(evidencia, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Evidencia: {(carpeta / 'evidencia.json').resolve()}")
    if filas:
        cv2.imwrite(str(carpeta / 'lamina.jpg'), np.vstack(filas))
        print(f"Lamina:    {(carpeta / 'lamina.jpg').resolve()}")
    print(f"Fotogramas: {evidencia['resumen']['fotogramas_analizados']} | limitaciones: {len(evidencia['limitaciones'])}")


if __name__ == '__main__':
    main()
