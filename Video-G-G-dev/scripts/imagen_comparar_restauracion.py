"""Experimento de la etapa 3: ¿restaurar la imagen ayuda a detectar?

Corre el detector sobre tres variantes de cada imagen y compara:
  original     -> la imagen tal como llega
  adaptativa   -> la mejora que ya existia (gamma / CLAHE / mediana)
  restaurada   -> adaptativa + restauracion nueva (ruido y enfoque)

Dos formas de uso, desde la carpeta Video-G-G-dev:

  1) Sin etiquetas, para mirar imagenes sueltas (cuenta lo que detecta cada variante):
     python -m scripts.imagen_comparar_restauracion --imagenes carpeta_con_fotos

  2) Con la hoja de etiquetas (mide contra lo que se anoto a mano):
     python -m scripts.imagen_comparar_restauracion --etiquetas datos_propios/etiquetas.csv

La forma 2 necesita el script del paso 2 (scripts/imagen_preparar_etiquetado.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

VARIANTES = ('original', 'adaptativa', 'restaurada')
EXTENSIONES = ('.jpg', '.jpeg', '.png', '.bmp')
# Calidad ficticia que dispara las dos operaciones de restaurar(), para --forzar-restauracion.
CALIDAD_FORZADA = {'ruido_mad_estimado': 1e9, 'nitidez_laplaciana': 0.0,
                   'desenfoque_movimiento': {'anisotropia': 0.0}}


def crear_variantes(original, *, forzar=False):
    """Devuelve ({nombre: imagen BGR}, bloque de calidad, bloque de restauracion)."""
    from app.modulo_imagen.nitidez import bloque_calidad
    from app.modulo_imagen.preprocesamiento import mejorar_imagen
    from app.modulo_imagen.restauracion import restaurar
    calidad = bloque_calidad(original)
    adaptativa = mejorar_imagen(original, modo='adaptativo')[0]
    restaurada, bloque = restaurar(adaptativa, CALIDAD_FORZADA if forzar else calidad)
    return {'original': original, 'adaptativa': adaptativa, 'restaurada': restaurada}, calidad, bloque


def detectar(imagen):
    """Lista de detecciones {'clase', 'confianza', 'bbox'} del detector del modulo."""
    from app.modulo_imagen.detector import detectar_imagen  # carga YOLO: solo al usarlo
    return detectar_imagen(imagen)[1]


def _xy(x, y):
    return f'{x} de {y} ({100.0 * x / y:.0f} %)' if y else f'{x} de {y}'


def modo_imagenes(carpeta, salida, forzar):
    import cv2
    rutas = sorted(p for p in Path(carpeta).iterdir() if p.suffix.lower() in EXTENSIONES)
    if not rutas:
        sys.exit(f'No hay imagenes en {carpeta}')
    lineas = ['# Comparacion de variantes sobre imagenes sueltas', '',
              'Cuenta lo que detecta cada variante. Sin etiquetas no se sabe cuantas detecciones son correctas.', '',
              '| Imagen | Nitidez | Alta frec. | Ruido | Anisotropia | Operaciones | Variante | Personas | Celulares | Otros objetos | ms |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for ruta in rutas:
        original = cv2.imread(str(ruta))
        if original is None:
            print(f'No se pudo leer {ruta.name}; se omite')
            continue
        variantes, calidad, bloque = crear_variantes(original, forzar=forzar)
        for nombre in VARIANTES:
            t0 = time.perf_counter()
            detecciones = detectar(variantes[nombre])
            ms = (time.perf_counter() - t0) * 1000
            clases = Counter(d['clase'] for d in detecciones)
            otros = ', '.join(f'{n} {c}' for c, n in sorted(clases.items()) if c not in ('person', 'cell phone')) or '-'
            lineas.append(f"| {ruta.name} | {calidad['nitidez_laplaciana']} | {calidad['energia_alta_frecuencia']} | "
                          f"{calidad['ruido_mad_estimado']} | {calidad['desenfoque_movimiento']['anisotropia']} | "
                          f"{', '.join(bloque['operaciones']) or 'ninguna'} | {nombre} | {clases['person']} | "
                          f"{clases['cell phone']} | {otros} | {ms:.0f} |")
            cv2.imwrite(str(salida / f'{ruta.stem}_{nombre}.jpg'), variantes[nombre])
    texto = '\n'.join(lineas) + '\n'
    (salida / 'comparacion_imagenes.md').write_text(texto, encoding='utf-8')
    print(texto)
    print(f'Resultados e imagenes de cada variante en: {salida.resolve()}')


def modo_etiquetas(ruta_etiquetas, carpeta_fotogramas, salida, forzar, histograma):
    import cv2
    try:
        from scripts.imagen_preparar_etiquetado import emparejar_por_iou, leer_etiquetas
    except ImportError:
        sys.exit('Falta scripts/imagen_preparar_etiquetado.py (paso 2 del plan). '
                 'Mientras tanto use --imagenes para probar con fotos sueltas.')
    filas = leer_etiquetas(ruta_etiquetas)
    por_fotograma = defaultdict(lambda: {'fotograma': None, 'personas': []})
    for fila in filas:
        clave = (fila['clip_id'], fila['fotograma'])
        if fila['tipo_fila'] == 'fotograma':
            por_fotograma[clave]['fotograma'] = fila
        else:
            por_fotograma[clave]['personas'].append(fila)
    cuenta = {v: {'pers_enc': 0, 'pers_real': 0, 'cel_enc': 0, 'cel_fot': 0, 'ms': 0.0,
                  'por_postura': defaultdict(lambda: [0, 0])} for v in VARIANTES}
    nitidez_por_calidad = defaultdict(list)
    usados = 0
    for (clip_id, nombre_foto), grupo in sorted(por_fotograma.items()):
        fila_foto = grupo['fotograma']
        if fila_foto is None or fila_foto.get('personas_reales') is None:
            continue  # fotograma aun sin contar a mano
        original = cv2.imread(str(Path(carpeta_fotogramas) / clip_id / nombre_foto))
        if original is None:
            print(f'No se pudo leer {clip_id}/{nombre_foto}; se omite')
            continue
        usados += 1
        variantes, calidad, _ = crear_variantes(original, forzar=forzar)
        etiqueta_calidad = fila_foto.get('calidad_visual') or next(
            (p.get('calidad_visual') for p in grupo['personas'] if p.get('calidad_visual')), None)
        if etiqueta_calidad:
            nitidez_por_calidad[etiqueta_calidad].append(calidad['nitidez_laplaciana'])
        reales = int(fila_foto['personas_reales'])
        celulares = int(fila_foto.get('celulares_reales') or 0)
        for nombre in VARIANTES:
            c = cuenta[nombre]
            t0 = time.perf_counter()
            detecciones = detectar(variantes[nombre])
            c['ms'] += (time.perf_counter() - t0) * 1000
            personas = [d for d in detecciones if d['clase'] == 'person']
            c['pers_enc'] += min(len(personas), reales)
            c['pers_real'] += reales
            if celulares > 0:
                c['cel_fot'] += 1
                c['cel_enc'] += int(any(d['clase'] == 'cell phone' for d in detecciones))
            encontradas = {id(f) for f in (emparejar_por_iou(d['bbox'], grupo['personas']) for d in personas) if f}
            for fila in grupo['personas']:
                postura = fila.get('postura') or 'sin_etiqueta'
                c['por_postura'][postura][1] += 1
                c['por_postura'][postura][0] += int(id(fila) in encontradas)
    if not usados:
        sys.exit('Ningun fotograma tiene `personas_reales` rellenado en la hoja.')
    posturas = sorted({p for v in VARIANTES for p in cuenta[v]['por_postura']})
    lineas = ['# Experimento: ¿restaurar ayuda a detectar?', '', f'Fotogramas usados: {usados}', '',
              '| Medida | ' + ' | '.join(VARIANTES) + ' |', '|---|---|---|---|',
              '| Personas encontradas | ' + ' | '.join(_xy(cuenta[v]['pers_enc'], cuenta[v]['pers_real']) for v in VARIANTES) + ' |',
              '| Fotogramas con celular encontrado | ' + ' | '.join(_xy(cuenta[v]['cel_enc'], cuenta[v]['cel_fot']) for v in VARIANTES) + ' |']
    for postura in posturas:
        lineas.append(f'| Personas etiquetadas recuperadas ({postura}) | ' + ' | '.join(
            _xy(*cuenta[v]['por_postura'][postura]) for v in VARIANTES) + ' |')
    lineas.append('| Tiempo medio de deteccion (ms) | ' + ' | '.join(f"{cuenta[v]['ms'] / usados:.0f}" for v in VARIANTES) + ' |')
    texto = '\n'.join(lineas) + '\n'
    (salida / 'comparacion.md').write_text(texto, encoding='utf-8')
    print(texto)
    if histograma:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, eje = plt.subplots(figsize=(7, 4))
        for etiqueta in ('nitida', 'borrosa'):
            if nitidez_por_calidad.get(etiqueta):
                eje.hist(nitidez_por_calidad[etiqueta], bins=20, alpha=0.6, label=f'{etiqueta} ({len(nitidez_por_calidad[etiqueta])})')
        eje.set_xlabel('nitidez_laplaciana del fotograma original')
        eje.set_ylabel('fotogramas')
        eje.set_title('Nitidez segun la etiqueta calidad_visual')
        eje.legend()
        fig.tight_layout()
        fig.savefig(salida / 'histograma_nitidez.png', dpi=120)
        print(f"Histograma: {(salida / 'histograma_nitidez.png').resolve()}")
    print(f'Resultados en: {salida.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='Compara la deteccion sobre original, mejora adaptativa y restaurada.')
    parser.add_argument('--imagenes', help='Carpeta con fotos sueltas (modo sin etiquetas)')
    parser.add_argument('--etiquetas', help='Ruta de etiquetas.csv (modo con etiquetas)')
    parser.add_argument('--fotogramas', default='datos_propios/fotogramas', help='Carpeta de fotogramas del modo con etiquetas')
    parser.add_argument('--salida', default='runs/imagen/restauracion')
    parser.add_argument('--forzar-restauracion', action='store_true',
                        help='Aplica ruido y enfoque aunque las medidas no lo pidan (para ver su efecto)')
    parser.add_argument('--histograma-nitidez', action='store_true',
                        help='Guarda el histograma de nitidez por calidad_visual (solo con --etiquetas)')
    args = parser.parse_args()
    if bool(args.imagenes) == bool(args.etiquetas):
        parser.error('Indique --imagenes o --etiquetas (uno de los dos)')
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if args.imagenes:
        modo_imagenes(args.imagenes, salida, args.forzar_restauracion)
    else:
        modo_etiquetas(args.etiquetas, args.fotogramas, salida, args.forzar_restauracion, args.histograma_nitidez)


if __name__ == '__main__':
    main()
