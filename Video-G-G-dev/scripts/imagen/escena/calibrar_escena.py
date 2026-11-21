"""Marca con clics la zona de escalones y los pasamanos de una camara fija.

Uso, desde la carpeta Video-G-G-dev:
    python -m scripts.imagen.escena.calibrar_escena foto_escalera_vacia.jpg --camara escalera_g

Controles de la ventana:
    clic izquierdo  anade un punto (primero las esquinas de la zona de escalones)
    p               pasa a marcar pasamanos: dos clics por cada uno (extremo y extremo).
                    Se pueden marcar varios seguidos (baranda central, baranda de pared).
    z               deshace el ultimo punto
    Enter           guarda configs/escena/<camara>.json
    Esc             sale sin guardar

El nombre de --camara debe ser exactamente el `camara_id` pactado con Video.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VENTANA = 'Calibrar escena'


def construir_calibracion(camara_id, ancho, alto, zona, pasamanos):
    """Arma el dict que se guarda. Lanza ValueError si lo marcado no alcanza.

    `pasamanos` es la lista de puntos marcados en orden: cada par es un pasamanos.
    Se guarda como lista de segmentos [[x1, y1, x2, y2], ...] o None.
    """
    if len(zona) < 3:
        raise ValueError('La zona de escalones necesita al menos 3 esquinas')
    if len(pasamanos) % 2:
        raise ValueError('Cada pasamanos necesita sus dos extremos: falta un punto')
    segmentos = [[int(pasamanos[i][0]), int(pasamanos[i][1]), int(pasamanos[i + 1][0]), int(pasamanos[i + 1][1])]
                 for i in range(0, len(pasamanos), 2)]
    return {'camara_id': camara_id, 'ancho_ref': int(ancho), 'alto_ref': int(alto),
            'zona_escalones': [[int(x), int(y)] for x, y in zona], 'pasamanos': segmentos or None}


def main():
    parser = argparse.ArgumentParser(description='Marca zona de escalones y pasamanos con clics.')
    parser.add_argument('foto', help='Foto de la escalera vacia, desde la posicion de la camara')
    parser.add_argument('--camara', required=True, help='camara_id pactado con Video (nombre del archivo)')
    parser.add_argument('--carpeta', default='configs/escena')
    args = parser.parse_args()

    import cv2
    import numpy as np
    foto = cv2.imread(args.foto)
    if foto is None:
        sys.exit(f'No se pudo leer la foto: {args.foto}')
    alto, ancho = foto.shape[:2]
    # La foto se muestra reducida si no cabe en pantalla; los puntos se guardan en pixeles reales.
    escala = min(1.0, 1280 / ancho, 720 / alto)
    estado = {'zona': [], 'pasamanos': [], 'modo': 'zona'}

    def al_clic(evento, x, y, *_):
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        punto = (int(round(x / escala)), int(round(y / escala)))
        if estado['modo'] == 'zona':
            estado['zona'].append(punto)
        else:
            estado['pasamanos'].append(punto)

    def dibujar():
        vista = cv2.resize(foto, None, fx=escala, fy=escala) if escala < 1 else foto.copy()
        a_vista = lambda p: (int(p[0] * escala), int(p[1] * escala))
        if estado['zona']:
            cv2.polylines(vista, [np.array([a_vista(p) for p in estado['zona']], dtype=np.int32)],
                          len(estado['zona']) >= 3, (60, 200, 60), 2)
        for p in estado['zona']:
            cv2.circle(vista, a_vista(p), 5, (60, 200, 60), -1)
        for i in range(0, len(estado['pasamanos']) - 1, 2):
            cv2.line(vista, a_vista(estado['pasamanos'][i]), a_vista(estado['pasamanos'][i + 1]), (40, 140, 255), 3)
        for p in estado['pasamanos']:
            cv2.circle(vista, a_vista(p), 5, (40, 140, 255), -1)
        texto = ('ZONA DE ESCALONES: clic en cada esquina | p = pasamanos' if estado['modo'] == 'zona'
                 else f"PASAMANOS {len(estado['pasamanos']) // 2 + 1}: clic en sus dos extremos (puede marcar varios)")
        cv2.putText(vista, texto + ' | z deshacer | Enter guardar | Esc salir', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return vista

    cv2.namedWindow(VENTANA)
    cv2.setMouseCallback(VENTANA, al_clic)
    while True:
        cv2.imshow(VENTANA, dibujar())
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == 27:                      # Esc
            print('Salida sin guardar.')
            break
        if tecla == ord('p'):
            estado['modo'] = 'pasamanos'
        elif tecla == ord('z'):
            lista = estado['pasamanos'] if estado['modo'] == 'pasamanos' and estado['pasamanos'] else estado['zona']
            if lista:
                lista.pop()
        elif tecla in (13, 10):              # Enter
            try:
                calibracion = construir_calibracion(args.camara, ancho, alto, estado['zona'], estado['pasamanos'])
            except ValueError as exc:
                print(f'No se guarda: {exc}')
                continue
            carpeta = Path(args.carpeta)
            carpeta.mkdir(parents=True, exist_ok=True)
            ruta = carpeta / f'{args.camara}.json'
            ruta.write_text(json.dumps(calibracion, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Calibracion guardada en {ruta.resolve()}')
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
