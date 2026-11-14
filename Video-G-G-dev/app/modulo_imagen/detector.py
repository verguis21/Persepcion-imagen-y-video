
import cv2
from ultralytics import YOLO

# Modelo preentrenado para detectar objetos
modelo = YOLO("yolo11n.pt")

def detectar_imagen(frame):
    resultados = modelo.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    imagen_anotada = resultados[0].plot()

    detecciones = []

    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre = modelo.names[clase_id]
        confianza = float(caja.conf[0])
        coordenadas = caja.xyxy[0].tolist()

        if nombre in ("person", "cell phone", "backpack", "handbag", "suitcase", "bottle", "cup", "chair"):
            detecciones.append({
                "clase": nombre,
                "confianza": round(confianza, 3),
                "bbox": [round(x, 2) for x in coordenadas]
            })

    return imagen_anotada, detecciones


if __name__ == "__main__":
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    print("Cámara iniciada. Presiona Q para salir.")

    try:
        while True:
            ok, frame = camara.read()

            if not ok:
                break

            imagen, objetos = detectar_imagen(frame)

            cv2.imshow("Modulo Imagen - YOLO", imagen)

            for objeto in objetos:
                print(objeto)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()


def buscar_celulares_en_manos(frame, personas, *, confianza_minima=0.23, max_recortes=4):
    """Segunda pasada exploratoria cerca de muñecas VISIBLES (solo COCO cell phone).

    Las detecciones no se inventan: solo se añaden cajas producidas por YOLO.
    Limitamos recortes y evitamos cambios de coordenadas. No garantiza reconocer
    celulares pequeños, borrosos u ocultos; aumenta cómputo y falsos positivos.
    """
    nombres = modelo.names
    pares = nombres.items() if isinstance(nombres, dict) else enumerate(nombres)
    id_cel = next((int(k) for k, n in pares if n == 'cell phone'), None)
    if id_cel is None:
        return []
    alto, ancho = frame.shape[:2]
    detectados, n = [], 0
    for persona in personas:
        pose = persona.get('pose') or {}
        puntos = pose.get('puntos_corporales') or []
        bbox = persona.get('bbox') or []
        if len(bbox) != 4:
            continue
        alto_persona = float(bbox[3]) - float(bbox[1])
        lado = int(max(96, min(260, alto_persona * 0.65)))
        for indice in (9, 10):  # ambas muñecas de YOLO11-Pose
            if n >= max_recortes:
                return detectados
            if len(puntos) <= indice or len(puntos[indice]) < 3 or puntos[indice][2] < 0.50:
                continue
            cx, cy = float(puntos[indice][0]), float(puntos[indice][1])
            x1 = max(0, int(round(cx - lado / 2)))
            y1 = max(0, int(round(cy - lado / 2)))
            x2 = min(ancho, x1 + lado)
            y2 = min(alto, y1 + lado)
            if x2 - x1 < 32 or y2 - y1 < 32:
                continue
            n += 1
            roi = frame[y1:y2, x1:x2]
            respuesta = modelo.predict(source=roi, classes=[id_cel], conf=confianza_minima,
                                      verbose=False)
            if not respuesta:
                continue
            for caja in respuesta[0].boxes:
                coords = caja.xyxy[0].tolist()
                confianza = float(caja.conf[0])
                total = [coords[0]+x1, coords[1]+y1, coords[2]+x1, coords[3]+y1]
                centro_x = (total[0]+total[2])/2
                centro_y = (total[1]+total[3])/2
                if ((centro_x - cx)**2 + (centro_y - cy)**2)**0.5 > lado * 0.40:
                    continue
                if any(_iou_cajas(total, d['bbox']) > 0.45 for d in detectados):
                    continue
                detectados.append({'clase':'cell phone', 'confianza':round(confianza,3),
                    'bbox':[round(v,2) for v in total], 'origen':'roi_mano_exploratorio'})
    return detectados


def _iou_cajas(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1])
    ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=area_a+area_b-inter
    return inter/union if union else 0.0
