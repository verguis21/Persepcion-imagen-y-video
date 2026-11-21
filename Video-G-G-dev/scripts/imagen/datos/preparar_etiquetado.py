"""Prepara el etiquetado manual (etapa 1 del plan).

1. Corta los videos largos en clips de hasta 10 s (los cortos se copian tal cual).
2. Saca fotogramas de cada clip (por defecto 2 por segundo).
3. Pasa cada fotograma por el detector y arma `etiquetas.csv`: una fila por
   fotograma (para contar a ojo personas y celulares) y una fila por cada
   persona detectada (para anotar postura, objeto en mano, pasamanos, calidad).

Uso, desde la carpeta Video-G-G-dev:
    python -m scripts.imagen.datos.preparar_etiquetado datos_propios/videos --fps 2

La UNIDAD que nunca se parte al entrenar es `video_origen`: dos pedazos del
mismo video son la misma persona, la misma luz y la misma ropa.

Las funciones auxiliares (`iou`, `leer_etiquetas`, `emparejar_por_iou`) no
cargan YOLO y las usan los scripts de restauracion, entrenamiento y evaluacion.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

COLUMNAS = ['clip_id', 'video_origen', 'fotograma', 'segundo_s', 'tipo_fila', 'detection_id', 'origen', 'bbox',
            'personas_reales', 'celulares_reales', 'postura', 'objeto_en_mano', 'toca_pasamanos',
            'calidad_visual', 'notas']
EXTENSIONES_VIDEO = ('.mp4', '.avi', '.mov', '.mkv')
MINIMO_VIDEOS = 8
PEDAZO_MINIMO_S = 2.0


# ---------------------------------------------------------------------------
# Auxiliares sin YOLO
# ---------------------------------------------------------------------------
def iou(a, b):
    """Interseccion sobre union de dos cajas [x1, y1, x2, y2]."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / union) if union > 0 else 0.0


def _entero(texto):
    texto = (texto or '').strip()
    return int(float(texto)) if texto else None


def leer_etiquetas(ruta):
    """Lee etiquetas.csv con tipos utiles. Las celdas vacias quedan en None."""
    filas = []
    with open(ruta, encoding='utf-8-sig', newline='') as f:
        for fila in csv.DictReader(f):
            fila = {k: (v.strip() if isinstance(v, str) else v) for k, v in fila.items() if k}
            fila['segundo_s'] = float(fila['segundo_s']) if fila.get('segundo_s') else None
            fila['bbox'] = [float(v) for v in json.loads(fila['bbox'])] if fila.get('bbox') else None
            for campo in ('detection_id', 'personas_reales', 'celulares_reales'):
                fila[campo] = _entero(fila.get(campo))
            for campo in ('postura', 'objeto_en_mano', 'toca_pasamanos', 'calidad_visual', 'notas', 'origen'):
                fila[campo] = fila.get(campo) or None
            if not fila.get('video_origen'):
                fila['video_origen'] = fila.get('clip_id')
            filas.append(fila)
    return filas


def emparejar_por_iou(bbox, filas, umbral=0.5):
    """Fila de tipo persona con mayor IoU >= umbral respecto a `bbox`, o None."""
    mejor, mejor_iou = None, umbral
    for fila in filas:
        if fila.get('tipo_fila') != 'persona' or not fila.get('bbox'):
            continue
        valor = iou(bbox, fila['bbox'])
        if valor >= mejor_iou:
            mejor, mejor_iou = fila, valor
    return mejor


# ---------------------------------------------------------------------------
# Videos -> clips -> fotogramas
# ---------------------------------------------------------------------------
def preparar_clips(carpeta_videos, carpeta_clips, segmento_s):
    """Devuelve [(ruta_clip, clip_id, video_origen)] y crea los clips cortados."""
    import cv2
    carpeta_clips.mkdir(parents=True, exist_ok=True)
    salida = []
    if not carpeta_videos.is_dir():
        for ruta in sorted(p for p in carpeta_clips.iterdir() if p.suffix.lower() in EXTENSIONES_VIDEO):
            salida.append((ruta, ruta.stem, ruta.stem))
        return salida
    for video in sorted(p for p in carpeta_videos.iterdir() if p.suffix.lower() in EXTENSIONES_VIDEO):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            cap.release()
            print(f'No se pudo leer {video.name}; se omite')
            continue
        duracion = total / fps
        if duracion <= segmento_s + 0.5:
            cap.release()
            destino = carpeta_clips / video.name
            if not destino.exists():
                shutil.copy2(video, destino)
            salida.append((destino, video.stem, video.stem))
            continue
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cuadros_por_pedazo = max(1, int(round(segmento_s * fps)))
        pedazo, escritor, ruta_actual, escritos = 0, None, None, 0
        pedazos = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if escritor is None:
                pedazo += 1
                ruta_actual = carpeta_clips / f'{video.stem}_s{pedazo:02d}.mp4'
                escritor = cv2.VideoWriter(str(ruta_actual), cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto))
                escritos = 0
            escritor.write(frame)
            escritos += 1
            if escritos >= cuadros_por_pedazo:
                escritor.release()
                pedazos.append((ruta_actual, escritos))
                escritor = None
        if escritor is not None:
            escritor.release()
            pedazos.append((ruta_actual, escritos))
        cap.release()
        for ruta, cuadros in pedazos:
            if cuadros / fps < PEDAZO_MINIMO_S:
                ruta.unlink(missing_ok=True)       # ultimo pedazo demasiado corto
                continue
            salida.append((ruta, ruta.stem, video.stem))
    return salida


def extraer_fotogramas(ruta_clip, fps_muestreo):
    """[(segundo_s, frame BGR)] tomando un fotograma cada 1/fps_muestreo segundos."""
    import cv2
    cap = cv2.VideoCapture(str(ruta_clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    salto = max(1, int(round(fps / fps_muestreo))) if fps > 0 else 1
    salida, indice = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if indice % salto == 0:
            salida.append((round(indice / fps, 3) if fps > 0 else float(len(salida)), frame))
        indice += 1
    cap.release()
    return salida


def _fila(**valores):
    fila = {c: '' for c in COLUMNAS}
    fila.update({k: ('' if v is None else v) for k, v in valores.items()})
    return fila


def main():
    parser = argparse.ArgumentParser(description='Corta videos, saca fotogramas y arma la hoja de etiquetas.')
    parser.add_argument('videos', nargs='?', default='datos_propios/videos', help='Carpeta con los videos originales')
    parser.add_argument('--segmento-s', type=float, default=10.0, help='Duracion maxima de cada clip')
    parser.add_argument('--fps', type=float, default=2.0, help='Fotogramas por segundo a extraer')
    parser.add_argument('--salida', default='datos_propios')
    parser.add_argument('--modo', default='adaptativo', help='Mejora previa a la deteccion (como en el modulo)')
    parser.add_argument('--sobrescribir', action='store_true', help='Reemplaza una etiquetas.csv existente')
    args = parser.parse_args()

    salida = Path(args.salida)
    ruta_csv = salida / 'etiquetas.csv'
    if ruta_csv.exists() and not args.sobrescribir:
        sys.exit(f'{ruta_csv} ya existe y puede tener etiquetas hechas a mano. '
                 'Use --sobrescribir solo si de verdad quiere reemplazarla.')
    import cv2
    from app.modulo_imagen.deteccion.analisis_tiempo_real import analizar_frame   # carga YOLO

    clips = preparar_clips(Path(args.videos), salida / 'clips', args.segmento_s)
    if not clips:
        sys.exit(f'No hay videos en {args.videos} ni clips en {salida / "clips"}')
    filas, n_fotogramas, n_personas = [], 0, 0
    for ruta, clip_id, video_origen in clips:
        carpeta = salida / 'fotogramas' / clip_id
        carpeta.mkdir(parents=True, exist_ok=True)
        for i, (segundo, frame) in enumerate(extraer_fotogramas(ruta, args.fps)):
            nombre = f'frame_{i:03d}.jpg'
            cv2.imwrite(str(carpeta / nombre), frame)
            dato, _ = analizar_frame(frame, modo=args.modo)
            n_fotogramas += 1
            comun = dict(clip_id=clip_id, video_origen=video_origen, fotograma=nombre, segundo_s=segundo)
            filas.append(_fila(tipo_fila='fotograma', **comun))
            for p in dato.get('personas') or []:
                filas.append(_fila(tipo_fila='persona', detection_id=p.get('detection_id'), origen='yolo',
                                   bbox=json.dumps([round(float(v), 1) for v in p['bbox']]), **comun))
                n_personas += 1
            for p in dato.get('poses_sin_asociar') or []:
                filas.append(_fila(tipo_fila='persona', origen='solo_pose',
                                   bbox=json.dumps([round(float(v), 1) for v in p['bbox']]), **comun))
                n_personas += 1
        print(f'{clip_id}: listo')
    salida.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel en Windows abre bien las tildes.
    with open(ruta_csv, 'w', encoding='utf-8-sig', newline='') as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    videos = sorted({v for _, _, v in clips})
    print(f'Videos originales: {len(videos)} | clips: {len(clips)} | fotogramas: {n_fotogramas} | '
          f'filas de persona: {n_personas}')
    print(f'Hoja: {ruta_csv.resolve()}')
    if len(videos) < MINIMO_VIDEOS:
        print(f'Aviso: hay {len(videos)} videos originales. Para entrenar y evaluar sin trampa conviene '
              f'tener al menos {MINIMO_VIDEOS}, con cada postura en dos o mas.')


if __name__ == '__main__':
    main()
