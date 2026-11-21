# Estructura del módulo Imagen

Los archivos de Imagen están agrupados por lo que hacen. Los de Video (`app/server.py`, `app/risk_engine.py`, `scripts/train.py`, `scripts/prepare_dataset.py`, `scripts/extract_frames.py`, etc.) no se movieron.

## Código del módulo: `app/modulo_imagen/`

Un fotograma recorre las carpetas en este orden: entrada → calidad → detección → escena → postura → salida.

| Carpeta | Para qué sirve | Archivos |
|---|---|---|
| `entrada/` | Recibir el clip de Video y elegir un fotograma por segundo | `contrato_video.py`, `analizador_clip.py` |
| `calidad/` | Medir la calidad, mejorar (gamma, CLAHE) y restaurar (ruido, enfoque) | `calidad.py`, `nitidez.py`, `preprocesamiento.py`, `restauracion.py` |
| `deteccion/` | Detectar personas y objetos (YOLO), esqueletos y analizar cada fotograma | `detector.py`, `pose.py`, `analizador.py`, `analizador_mejorado.py`, `analisis_tiempo_real.py`, `recuperacion_roi.py`, `pose_roi.py`, `continuidad_visual.py` |
| `escena/` | Zona de escalones y pasamanos; objetos en mano y contacto con el pasamanos | `escena.py`, `relaciones.py` |
| `postura/` | Clasificador de postura propio (HOG + LBP + SVM) | `descriptores.py`, `clasificador_postura.py` |
| `salida/` | Evidencia para Predicción, señales del laboratorio y características para Video | `evidencia.py`, `senalizacion.py`, `features.py` |

## Páginas y rutas web: `app/imagen_web/`

| Archivo | Qué es | Cómo se abre |
|---|---|---|
| `servidor.py` | Servidor de Imagen con la ruta para Video (`/api/imagen/v1/...`) y el laboratorio | `python -m app.imagen_web.servidor` |
| `integracion.py` | Rutas `/api/imagen/v1/clip` y `/api/imagen/v1/evidencia` | (lo monta `servidor.py`) |
| `visor_pipeline.py` | Visor con un botón por etapa | `python -m app.imagen_web.visor_pipeline` |
| `laboratorio.py`, `pruebas.py` | Páginas de prueba | (las monta `servidor.py`) |
| `servidor_pruebas.py` | Servidor mínimo solo con la página de pruebas | `python -m app.imagen_web.servidor_pruebas` |

Las plantillas siguen en `app/templates/` y los estilos en `app/static/`.

## Scripts: `scripts/imagen/`

Todos se ejecutan desde `Video-G-G-dev`, con el entorno activo.

| Carpeta | Script | Comando |
|---|---|---|
| `datos/` | Cortar videos, sacar fotogramas y armar la hoja de etiquetas | `python -m scripts.imagen.datos.preparar_etiquetado datos_propios/videos --fps 2` |
| `calidad/` | Experimento de restauración (¿ayuda a detectar?) | `python -m scripts.imagen.calidad.comparar_restauracion --etiquetas datos_propios/etiquetas.csv` |
| `calidad/` | Comparar calidad y mejora adaptativa de una imagen | `python -m scripts.imagen.calidad.comparar_calidad foto.jpg` |
| `escena/` | Marcar con clics la zona de escalones y los pasamanos | `python -m scripts.imagen.escena.calibrar_escena foto.jpg --camara escalera_g` |
| `postura/` | Entrenar el clasificador de postura | `python -m scripts.imagen.postura.entrenar_postura --etiquetas datos_propios/etiquetas.csv` |
| `demo/` | Evidencia y lámina de un clip | `python -m scripts.imagen.demo.demo clip.mp4 --camara escalera_g` |
| `demo/` | Pruebas sueltas (`probar_clip.py`, `probar_celular.py`, `probar_roi.py`, `probar_integracion.py`) | `python -m scripts.imagen.demo.probar_clip ...` |

## Pruebas: `tests/imagen/`

`python -m pytest tests -q` corre todas. El instalador `instalar_imagen.ps1` no cambió.

## Nombres anteriores

| Antes | Ahora |
|---|---|
| `app/modulo_imagen/<archivo>.py` | `app/modulo_imagen/<carpeta>/<archivo>.py` (tabla de arriba) |
| `app/servidor_imagen_integrado.py` | `app/imagen_web/servidor.py` |
| `app/integracion_imagen.py` | `app/imagen_web/integracion.py` |
| `app/pipeline_imagen.py` | `app/imagen_web/visor_pipeline.py` |
| `app/laboratorio_imagen.py` | `app/imagen_web/laboratorio.py` |
| `app/imagen_pruebas.py` | `app/imagen_web/pruebas.py` |
| `app/pruebas_imagen_local.py` | `app/imagen_web/servidor_pruebas.py` |
| `scripts/imagen_preparar_etiquetado.py` | `scripts/imagen/datos/preparar_etiquetado.py` |
| `scripts/imagen_comparar_restauracion.py` | `scripts/imagen/calidad/comparar_restauracion.py` |
| `scripts/comparar_calidad.py`, `scripts/comparar_adaptativo.py` | `scripts/imagen/calidad/` |
| `scripts/imagen_calibrar_escena.py` | `scripts/imagen/escena/calibrar_escena.py` |
| `scripts/imagen_entrenar_postura.py` | `scripts/imagen/postura/entrenar_postura.py` |
| `scripts/imagen_demo.py` | `scripts/imagen/demo/demo.py` |
| `scripts/probar_clip_imagen.py`, `probar_celular.py`, `probar_roi.py`, `probar_integracion_video_imagen.py` | `scripts/imagen/demo/` |
| `tests/test_*.py`, `tests/fixtures/` | `tests/imagen/` |

Único cambio en un archivo de Video: en `app/server.py`, la línea `from modulo_imagen.features import build_features` pasó a `from modulo_imagen.salida.features import build_features`.
