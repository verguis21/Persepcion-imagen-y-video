"""Prueba REAL por HTTP: Video 5000 -> Imagen 5051 con el MISMO fotograma.

Desde raiz del proyecto:
  python -m scripts.imagen.demo.probar_integracion --imagen pruebas/ejemplo.jpg

El servidor Video debe estar arrancado si no se pasa --solo-imagen.
El token del panel de Video, si esta activado, se lee de PANEL_TOKEN.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
from app.modulo_imagen.entrada.contrato_video import contexto_desde_video


def main():
    ap = argparse.ArgumentParser(description='Intercambio Video -> Imagen (sin tocar RiskEngine)')
    origen = ap.add_mutually_exclusive_group(required=True)
    origen.add_argument('--imagen', help='Fotografia JPEG/PNG tomada del video original')
    origen.add_argument('--clip', help='Clip de 10.5 s maximo, para Imagen (sin inferencia previa de Video)')
    ap.add_argument('--video-url', default='http://127.0.0.1:5000')
    ap.add_argument('--imagen-url', default='http://127.0.0.1:5051')
    ap.add_argument('--frame-id', default='prueba_001')
    ap.add_argument('--clip-id', default='clip_prueba_001')
    ap.add_argument('--intervalo-s', type=float, default=1.0)
    ap.add_argument('--camara-id', default='camara_prueba')
    ap.add_argument('--tiempo-s', type=float, default=0.0)
    ap.add_argument('--solo-imagen', action='store_true', help='No llamar al detector Video')
    ap.add_argument('--anotada', action='store_true', help='Pedir JPEG anotado base64 en respuesta')
    ap.add_argument('--salida', default='runs/imagen/integracion_video_imagen.json')
    args = ap.parse_args()

    p = Path(args.imagen or args.clip)
    if not p.is_file():
        ap.error(f'No existe el archivo: {p}')
    limite = 50 if args.clip else 7
    if p.stat().st_size > limite * 1024 * 1024:
        ap.error(f'El archivo supera {limite}MB')
    raw = p.read_bytes()
    headers = {}
    if os.environ.get('PANEL_TOKEN'):
        headers['X-Panel-Token'] = os.environ['PANEL_TOKEN']

    try:
        if args.clip:
            metadata = {'clip_id': args.clip_id, 'camara_id': args.camara_id,
                        'tiempo_s': args.tiempo_s, 'contexto_video': {}}
            print('Clip: Imagen analizara los fotogramas; este modo no consulta Video.')
            r = requests.post(args.imagen_url.rstrip('/') + '/api/imagen/v1/clip',
                              files={'clip': (p.name, raw, 'application/octet-stream')},
                              data={'metadatos': json.dumps(metadata, ensure_ascii=False),
                                    'intervalo_s': str(args.intervalo_s)},
                              timeout=240)
        else:
            r = _analizar_fotograma(args, p, raw, headers)
        r.raise_for_status()
        resultado = r.json()
        archivo = Path(args.salida)
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding='utf-8')
        if args.clip:
            clip = resultado.get('resultado_clip', {})
            print(f"Imagen: {clip.get('fotogramas_analizados', 0)} fotogramas analizados")
        else:
            analiza = resultado.get('resultado_imagen', {})
            print(f"Imagen: {len(analiza.get('personas', []))} persona(s), "
                  f"{len(analiza.get('poses_sin_asociar', []))} pose(s) sin asociar")
        print(f'Respuesta guardada en: {archivo.resolve()}')
    except requests.RequestException as exc:
        raise SystemExit(f'Error HTTP: {exc}. Comprueba ambos servidores, puertos y PANEL_TOKEN.') from exc
    except (ValueError, KeyError) as exc:
        raise SystemExit(f'Error de contrato: {exc}') from exc


def _analizar_fotograma(args, p, raw, headers):
    if args.solo_imagen:
        metadata = {'frame_id': args.frame_id, 'camara_id': args.camara_id,
                    'tiempo_s': args.tiempo_s, 'contexto_video': {}}
        print('Modo aislado: Imagen recibira frame sin detecciones de Video.')
    else:
        v = requests.post(args.video_url.rstrip('/') + '/api/detect',
                          files={'file': (p.name, raw, 'application/octet-stream')},
                          headers=headers, timeout=180)
        v.raise_for_status()
        salida_video = v.json()
        metadata = contexto_desde_video(salida_video, frame_id=args.frame_id,
                                       camara_id=args.camara_id, tiempo_s=args.tiempo_s)
        print(f"Video: {len(metadata['contexto_video']['escaleras'])} escalera(s), "
              f"{len(metadata['contexto_video']['tracks'])} track(s)")

    return requests.post(args.imagen_url.rstrip('/') + '/api/imagen/v1/fotograma',
                         files={'frame': (p.name, raw, 'application/octet-stream')},
                         data={'metadatos': json.dumps(metadata, ensure_ascii=False),
                               'incluir_anotada': '1' if args.anotada else '0'},
                         timeout=180)


if __name__ == '__main__':
    main()
