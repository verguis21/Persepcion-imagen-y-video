# Plan: módulo Imagen como "segunda mirada" que entrega evidencia

Fecha: 2026-11-14 · Proyecto: detección de accidentes en escaleras universitarias · Subsistema: Imagen · Carpeta de trabajo: `Video-G-G-dev/` (todas las rutas de este plan son relativas a ella).

## 1. Planteamiento

El sistema completo tiene cuatro subsistemas: Video, Imagen, Audio y Predicción. Video vigila la escalera en continuo y, cuando ve algo raro (tropiezo, objeto indebido, postura extraña), recorta un **clip de riesgo** de hasta 10 segundos. Imagen recibe ese clip y lo trata como una serie de fotogramas repartidos en el tiempo. Predicción decide.

Hoy Imagen y Video se pisan: los dos detectan personas y posturas con redes preentrenadas. Este plan le da a Imagen un papel propio: es la **segunda mirada, lenta y detallista**, sobre pocos fotogramas. Mide la calidad de cada fotograma, lo restaura si conviene, describe la escena (escalones, pasamanos, objetos, qué lleva cada persona en las manos, cómo está su cuerpo) y entrega todo eso como **evidencia medible**.

Regla central, acordada: **Imagen entrega evidencia, no veredictos.** Ningún campo de salida se llama riesgo, probabilidad de accidente ni alerta. Imagen dice "tronco inclinado 40°, celular cerca de la mano derecha, muñeca lejos del pasamanos, fotograma con borrón de movimiento"; Predicción decide qué significa. La única cifra tipo porcentaje que se conserva es la confianza que ya devuelve el detector, renombrada `confianza_detector` para que no se confunda con riesgo.

Segunda regla: **el trabajo se hace a mano donde se aprende.** Cada etapa indica qué hace la persona y qué puede asistir una IA, y está atada a la sesión del curso que la sustenta, para que cada integrante pueda defender su pieza.

### Decisiones cerradas el 14/11 (tras la revisión del plan)

1. **Detección doble.** Si el experimento de la etapa 3 muestra que restaurar ayuda, el detector corre dos veces por fotograma, sobre la imagen actual y sobre la restaurada, y las dos salidas viajan como evidencia (campo `deteccion_restaurada`). Se decide con los números del experimento, a más tardar el miércoles 25/11.
2. **Privacidad en lo mínimo, por ser un trabajo académico.** Código de actor (`a1`, `a2`) en vez de nombre en los archivos, e imágenes fuera de git. No se hace tratamiento especial si pasa un tercero; se declara como limitación.
3. **`camara_id` pactado con Video.** Es un dato obligatorio del contrato: el mismo texto exacto que se usa al calibrar.
4. **Versiones fijadas:** `scikit-learn==1.9.1`, `scikit-image==0.26.0`, `joblib==1.6.0`.
5. **Instalador único** `instalar_imagen.ps1`, que sirve desde las primeras pruebas hasta la máquina de la demo. Ningún paquete nuevo entra sin su línea en `requirements-imagen.txt`.
6. **Se aprovechan los videos largos que ya hay**, cortados en clips de hasta 10 segundos. Para entrenar y evaluar, la unidad que no se parte es el **video original**, no el pedazo.

### Qué existe ya y qué falta

| Ya existe en `app/modulo_imagen/` | Falta |
|---|---|
| Métricas de calidad (`calidad.py`): luminosidad, contraste, nitidez por Laplaciano, ruido | Medir nitidez en frecuencia (S09) y distinguir desenfoque de movimiento |
| Mejora adaptativa (`preprocesamiento.py`): gamma, CLAHE, mediana | Restauración (S10) y comprobar con datos si mejorar la imagen mejora la detección |
| Objetos COCO con YOLO11 (`detector.py`) y pose (`pose.py`) | Describir la escena: escalones, pasamanos, objetos sobre escalones (S05) |
| Regla de postura por ángulo (`analisis_tiempo_real._evaluar_postura`) | Un clasificador propio entrenado con sus escaleras (S06, S11) |
| Muestreo de clip y contrato con Video (`analizador_clip.py`, `contrato_video.py`) | Un contrato de salida para Predicción |
| Pruebas unitarias con imágenes sintéticas (`tests/`) | **Datos propios etiquetados y métricas medidas**: hoy no hay ninguna |

### Calendario real

Según el sílabo (inicio 31/08/2026), la semana 12 empieza el lunes 16/11 y la **defensa del proyecto final es en la semana 14 (30/11 al 04/12)**. Quedan dos semanas de trabajo. El plan está dimensionado para eso: un núcleo obligatorio pequeño y extensiones opcionales marcadas como tales. Confirmar con el docente el día exacto de la exposición.

## 2. Resumen de etapas

| Etapa | Qué se hace | Sesión del curso | Responsable | Producto |
|---|---|---|---|---|
| 0. Contrato | Pactar el JSON de evidencia con Predicción y Video | S01 (flujo de percepción) | Los tres | `evidencia_ejemplo.json` entregado a Predicción |
| 1. Adquisición y etiquetado | Grabar clips propios, extraer fotogramas, etiquetar en hoja | S02 (muestreo), S03 (calidad de datos) | C lidera, graban los tres | Clips + `etiquetas.csv` |
| 2. Calidad del fotograma | Medir nitidez, luz, ruido y borrón; no descartar | S02, S03, S09 (Fourier) | A | Bloque `calidad` |
| 3. Restauración | Reducir ruido y realzar; medir si ayuda a detectar | S03, S04, S10 | A | Bloque `restauracion` + experimento |
| 4. Escena | Zona de escalones, pasamanos, objetos sobre escalones | S04 (geometría), S05 (segmentación) | B | Bloque `escena` |
| 5. Personas y objetos | Objetos en mano, contacto con pasamanos | S06 (geometría de puntos) | B | Campos por persona |
| 6. Postura propia | HOG + LBP y un SVM entrenado por ustedes, comparado con la regla de pose | S06, S11 | C | Bloque `postura` + modelo |
| 7. Evaluación | Matriz de confusión y recall por campo, con clips que el modelo no vio | S11 | C, con datos de A y B | `informe.md` |
| 8. Integración y demo | Unir bloques en el JSON, ruta HTTP, guion de demostración | S01, S15 (ética) | A integra, exponen los tres | Demo de semana 14 |

Las etapas 2 a 6 se trabajan **en paralelo** desde el día 2. La etapa 0 y la grabación de la etapa 1 son lo único que va primero.

## 3. Cómo se reparte sin que nadie espere a nadie

Tres decisiones hacen independientes las piezas.

1. **Un mismo lote de fotogramas para todos.** En cuanto se graban los clips (día 2), la carpeta `datos_propios/` se comparte por Drive. Cada persona prueba su pieza sobre esos fotogramas, sin necesitar el código de los demás.
2. **Cada pieza es una función pura con su propia entrada y salida.** Recibe un fotograma BGR (y, si hace falta, el diccionario `dato` que ya produce `analizar_frame`) y devuelve un diccionario: su bloque del JSON. Cada pieza vive en su archivo, con su prueba y su script de demostración.
3. **El contrato admite `null`.** `evidencia.py` arma el JSON con los bloques que existan; si una pieza aún no está, su campo va en `null`. Predicción recibe desde el día 2 un JSON de ejemplo con la forma final.

| Persona | Rol | Archivos propios | Qué aprende y defiende |
|---|---|---|---|
| A | Calidad y restauración; integra al final | `nitidez.py`, `restauracion.py`, `evidencia.py` | Muestreo, ruido, filtros espaciales y en frecuencia, Fourier |
| B | Escena y objetos | `escena.py`, `relaciones.py`, `scripts/imagen_calibrar_escena.py` | Bordes, Hough, morfología, segmentación, geometría |
| C | Datos, postura y evaluación | `descriptores.py`, `clasificador_postura.py`, scripts de etiquetado, entrenamiento y evaluación | Descriptores, pipeline de ML, división sin fuga, métricas |

**Si son dos personas:** A se queda con las etapas 2, 3 y 8; la otra persona toma 1, 6 y 7. La etapa 4 se reduce a la calibración manual (sin Hough ni resta de fondo) y la 5 a "objetos en mano", que ya está casi hecha en el código actual.

### Qué hace la persona y qué puede asistir la IA

| Tarea | La hace una persona | Puede asistir una IA |
|---|---|---|
| Decidir qué situaciones grabar y grabarlas | Sí, siempre | No |
| Etiquetar fotogramas en la hoja | Sí, siempre; es donde se entiende el dato | No (solo generar la plantilla vacía) |
| Elegir umbrales (nitidez, distancia al pasamanos) | Sí, mirando histogramas de sus propios datos | Graficar los histogramas |
| Marcar la zona de escalones y el pasamanos | Sí, con clics sobre una foto | Escribir el script de clics |
| Escribir las funciones de cada etapa | La persona escribe o, como mínimo, explica línea por línea antes de integrar | Borrador de código, pruebas unitarias, depuración |
| Leer la matriz de confusión y decir qué falla | Sí | Generar la figura |
| Redactar limitaciones y conclusiones | Sí | Revisar redacción |

Regla de equipo: **nada entra a `master` si su autor no puede explicarlo sin leer.** Cada etapa cierra con una "pregunta de defensa" (sección 5) que el responsable debe poder contestar.

## 4. El contrato con Predicción: `imagen-evidencia-1.0`

Se pacta el lunes 16/11 y se entrega el martes 17/11 un archivo de ejemplo, aunque sus valores sean inventados. Predicción programa contra esa forma mientras Imagen rellena campos. Regla de compatibilidad: **se pueden añadir campos; no se renombra ni se borra ninguno** sin subir la versión.

```json
{
  "schema_version": "imagen-evidencia-1.0",
  "clip": {"clip_id": "g2_1117_a1_celular_02", "camara_id": "escalera_pabellon_g", "duracion_s": 8.4, "fps_origen": 30.0, "contexto_video": {"escaleras": [], "tracks": []}},
  "fotogramas": [
    {
      "indice": 0,
      "segundo_s": 0.0,
      "calidad": {"luminosidad": 96.2, "contraste": 41.0, "nitidez_laplaciana": 212.5, "energia_alta_frecuencia": 0.18, "ruido_mad_estimado": 2.0, "desenfoque_movimiento": {"anisotropia": 0.41, "direccion_grados": 12.0}, "apto_para_detalle": true},
      "restauracion": {"operaciones": ["nlm_h=6", "mascara_enfoque=0.8"], "nitidez_antes": 212.5, "nitidez_despues": 301.9},
      "deteccion_restaurada": {"personas": 1, "objetos": 2, "solo_tras_restaurar": [{"clase": "cell phone", "bbox": [301, 262, 322, 300], "confianza_detector": 0.41}]},
      "escena": {"origen_zona": "calibracion_manual", "zona_escalones": [[120, 400], [520, 380], [600, 700], [60, 710]], "lineas_escalon_detectadas": 7, "pasamanos": {"visible": true, "segmento": [540, 120, 640, 620]}},
      "personas": [
        {"detection_id": 0, "origen": "yolo", "bbox": [210, 140, 330, 520], "confianza_detector": 0.88, "inclinacion_torso_grados": 38.0, "postura_reglas_pose": "inclinacion_para_revision", "postura_clasificador": "inclinada", "concordancia_posturas": true, "en_zona_escalones": true, "contacto_pasamanos": "no", "objetos_en_mano": [{"clase": "cell phone", "mano": "derecha", "confianza_detector": 0.47}]}
      ],
      "objetos": [{"clase": "backpack", "bbox": [400, 600, 470, 680], "confianza_detector": 0.61, "en_zona_escalones": true, "asociado_a_persona": null}]
    }
  ],
  "resumen": {"fotogramas_analizados": 9, "personas_max": 1, "fotogramas_con_desenfoque": 2, "posturas_observadas": {"erguida": 6, "inclinada": 3}, "clases_objeto": {"cell phone": 4, "backpack": 9}},
  "limitaciones": ["Clasificador de postura entrenado con N fotogramas de M clips propios; ver informe de evaluación."]
}
```

Valores permitidos: `postura_clasificador` ∈ {`erguida`, `inclinada`, `agachada_o_sentada`, `horizontal`, `no_evaluable`, `no_disponible`}; `contacto_pasamanos` ∈ {`si`, `no`, `no_evaluable`}; `origen_zona` ∈ {`calibracion_manual`, `no_disponible`}; `mano` ∈ {`izquierda`, `derecha`} (mano anatómica de la persona según los puntos COCO 9 = muñeca izquierda y 10 = muñeca derecha, no el lado de la imagen); `postura_reglas_pose` ∈ {`sin_indicio_suficiente`, `inclinacion_para_revision`, `postura_horizontal_posible`, `no_evaluable`} (los valores que ya produce `_evaluar_postura`); `origen` ∈ {`yolo`, `solo_pose`, `roi_yolo`, `roi_pose`}. `apto_para_detalle`, `en_zona_escalones`, `concordancia_posturas`, `detection_id` y `confianza_detector` (en personas que no vienen de YOLO) y cualquier bloque aún no implementado pueden ser `null`.

**Qué personas entran en `personas`.** No solo `dato['personas']` (cajas YOLO con pose asociada o no): también cada elemento de `dato['poses_sin_asociar']` (`origen: "solo_pose"`) y la persona de `dato['recuperacion_roi']` cuando tiene `origen` (`roi_yolo` o `roi_pose`). Son justo los casos en que YOLO no ve a alguien recostado o agachado, y perderlos dejaría a Predicción sin la evidencia más importante. En esas entradas `detection_id` y, si no hay caja YOLO, `confianza_detector` van en `null`.

**`deteccion_restaurada`** es `null` mientras la detección doble esté apagada. Cuando está encendida, resume la segunda pasada del detector sobre el fotograma restaurado: cuántas personas y objetos encontró, y en `solo_tras_restaurar` las detecciones que no coinciden (misma clase e IoU ≥ 0,5) con ninguna de la pasada normal. Las listas `personas` y `objetos` del fotograma siguen saliendo de la pasada normal; lo que aparece solo tras restaurar se reporta aparte para que Predicción sepa de dónde viene.

**`contexto_video` no se copia entero.** `normalizar_metadatos` siempre añade `contexto_video.riesgo_reportado_por_video` (aunque valga `null`). Esa clave contiene "riesgo" y además es una valoración de Video, no evidencia de Imagen: `construir_evidencia` copia solo `escaleras` y `tracks`; Predicción recibe el riesgo de Video directamente de Video. Por la misma razón no se copia `resumen_visual` (contiene `fotogramas_alerta_seguimiento`) ni `senales_visuales`.

**Correspondencia regla ↔ clasificador** (para `concordancia_posturas` y para la matriz de confusión de la regla en la etapa 7): `sin_indicio_suficiente` ↔ `erguida`; `inclinacion_para_revision` ↔ `inclinada`; `postura_horizontal_posible` ↔ `horizontal`. La regla no tiene equivalente de `agachada_o_sentada`: si el clasificador dice esa clase, o cualquiera de los dos dice `no_evaluable`/`no_disponible`, `concordancia_posturas` es `null`. En la evaluación, la regla nunca acierta `agachada_o_sentada` y eso se reporta como limitación de la línea base.

**Versión mínima a pactar el día 1** (lo demás llega después como campos nuevos): `clip.clip_id`, y por fotograma `segundo_s`, `calidad.nitidez_laplaciana`, y por persona `bbox`, `postura_reglas_pose`, `objetos_en_mano`. Todo eso ya lo calcula el código actual.

Lo que el contrato **no** lleva: los títulos de `senalizacion.py` ("ATENCIÓN PRIORITARIA", "ALERTA EN SEGUIMIENTO"). Esas señales siguen existiendo para el laboratorio web, pero son interpretación y no viajan a Predicción.

## 5. Etapas en detalle

### Etapa 0. Contrato (lunes 16 y martes 17/11, los tres)

- Reunión corta con Predicción y Video con el JSON de la sección 4 impreso. Preguntas a cerrar: ¿qué campos usará Predicción de verdad?, ¿cuántos fotogramas por clip esperan (hoy: uno por segundo)? **Acuerdo obligatorio con Video:** que envíe siempre `clip_id` y `camara_id`, y cuál es el texto exacto de `camara_id` para cada escalera (por ejemplo `escalera_g`). Ese mismo texto es el nombre del archivo de calibración `configs/escena/<camara_id>.json`; si no coinciden, la zona de escalones y el pasamanos salen vacíos. Se anota en una tabla compartida de definiciones comunes (identificadores de cámara, formato de `clip_id`, duración máxima del clip).
- A escribe `app/modulo_imagen/evidencia.py` con `validar_evidencia` y el archivo `tests/fixtures/evidencia_ejemplo.json`, y lo envía.

Resultado: contrato pactado y ejemplo entregado. **Pregunta de defensa:** ¿por qué Imagen no entrega un porcentaje de riesgo?

### Etapa 1. Adquisición y etiquetado (grabar 16–17/11, etiquetar 18–22/11; C lidera)

Conceptos del curso desde el inicio: grabar ya es **muestrear** (S02). Un clip a 30 fps y 1080p son 30 muestras por segundo en el tiempo y unos 2 millones en el espacio; al tomar un fotograma por segundo se pierde lo que pasa entre medias, y un movimiento rápido se convierte en borrón porque el obturador integra luz mientras el cuerpo se desplaza. Esa es la razón técnica de la etapa 2.

- **Qué grabar** (situaciones simuladas y seguras; **nunca caídas reales ni actuadas sobre los escalones**): subir y bajar normal; con celular en la mano mirando la pantalla; con mochila; con las dos manos ocupadas; agarrado del pasamanos y sin agarrarse; agacharse a recoger algo; sentado en un escalón; inclinado hacia delante con apoyo en el pasamanos; un objeto dejado sobre un escalón sin nadie; dos personas cruzándose; paso rápido (para provocar borrón); poca luz o contraluz. La postura `horizontal` se graba **solo en el descanso plano**, recostado en el suelo.
- **Cuánto:** lo ideal son 50 a 70 clips de 5 a 10 segundos (el analizador rechaza clips de más de 10,5 s). **Si ya hay videos largos, se aprovechan:** se dejan en `datos_propios/videos/` y el script de extracción los corta en clips de hasta 10 segundos. Pero dos pedazos del mismo video son la misma persona, la misma luz y la misma ropa: para entrenar y evaluar cuentan como **un solo grupo** (columna `video_origen`). Lo que importa es cuántos videos originales distintos hay: hacen falta al menos unos 8 a 10, y que cada postura aparezca en 2 o más. Si una postura falta o sale en un solo video, se graban clips cortos nuevos de esa postura; el aumento de datos (volteo, brillo) da variedad pero no inventa posturas que nadie grabó. Con celular en horizontal, apoyado fijo donde iría la cámara. Si se puede, **dos escaleras o dos días distintos**: una queda reservada para prueba.
- **Nombre de archivo:** `<escalera>_<fecha>_<actor>_<situacion>_<toma>.mp4`, por ejemplo `g2_1117_a1_celular_02.mp4`, donde `<actor>` es un **código** (`a1`, `a2`), no el nombre de la persona; la tabla de quién es quién se queda en el Drive. El nombre es el `clip_id` y se fija al grabar, porque lo heredan los fotogramas, la hoja, el entrenamiento y el JSON. Los pedazos de un video largo se llaman `<video>_s01`, `<video>_s02`, etc.
- **Extracción:** `python -m scripts.imagen_preparar_etiquetado datos_propios/videos --fps 2` corta primero los videos de más de 10 segundos en clips dentro de `datos_propios/clips/` (los cortos se copian tal cual), y guarda los fotogramas en `datos_propios/fotogramas/<clip_id>/` y crea `datos_propios/etiquetas.csv` con una fila por persona detectada y las columnas de etiqueta vacías. Cada fila guarda `clip_id`, `video_origen`, `fotograma` (nombre del archivo), `segundo_s`, `detection_id`, `origen` y la `bbox` del detector, porque los scripts de las etapas 3, 6 y 7 emparejan sus propias detecciones con la fila por IoU ≥ 0,5 en el mismo fotograma (el `detection_id` cambia si cambia el modo de preprocesamiento). Además, cada fotograma lleva **una fila de fotograma** (`detection_id` vacío) con la columna `personas_reales` (cuántas personas se ven a ojo) y `celulares_reales`: sin ella, la plantilla solo contiene lo que el detector ya encontró y no se podría medir a quién se le escapa (etapa 3) ni el recall del celular (etapa 7). Se puede partir de la lógica de extracción por clip de `scripts/extract_frames.py`.
- **Etiquetado a mano** en Excel o Google Sheets, sin cajas (la caja ya la da el detector). Columnas a llenar: `postura` (erguida / inclinada / agachada_o_sentada / horizontal / no_se_ve), `objeto_en_mano` (ninguno / celular / otro), `toca_pasamanos` (si / no / no_se_ve), `calidad_visual` (nitida / borrosa / oscura), `notas`. Meta: 300 a 400 filas etiquetadas. Es trabajo de una tarde larga.
- **Control de calidad del etiquetado** (S03): una segunda persona etiqueta a ciegas 40 filas; se cuenta en cuántas coinciden. Si la coincidencia en `postura` baja de 80 %, la definición de las clases está confusa y se reescribe antes de seguir.
- **Privacidad y ética** (S15), en lo mínimo que corresponde a un trabajo académico: código de actor en vez de nombre; los videos, clips y fotogramas **no se suben a git** (se comparten por Drive); `etiquetas.csv` sí se versiona porque no contiene imágenes ni nombres. Si en un video aparece alguien ajeno al equipo no se hace ningún tratamiento especial; se declara como limitación en la exposición, junto con que un despliegue real exigiría aviso, consentimiento y difuminado de rostros.

Resultado: clips, fotogramas y `etiquetas.csv`. **Pregunta de defensa:** ¿por qué se separa por video original y no por fotograma?

### Etapa 2. Calidad del fotograma (A, S02 + S03 + S09)

Decisión acordada: **los fotogramas borrosos no se descartan; se miden.** Un borrón de movimiento indica velocidad y es evidencia. `apto_para_detalle` solo avisa a las etapas siguientes y a Predicción de cuánto fiarse de lo fino (un celular en la mano).

- Se reutiliza `medir_calidad` tal cual (Laplaciano, luminosidad, contraste, ruido).
- Nuevo `nitidez.py`: `energia_alta_frecuencia` con la FFT 2D (S09): proporción de la energía del espectro fuera de un disco central de radio 10 % del lado menor. Una imagen desenfocada concentra la energía en el centro.
- Nuevo `desenfoque_movimiento`: se comparan las energías de los gradientes de Sobel en X y en Y. Un borrón horizontal borra los bordes verticales, así que la energía en X cae frente a Y. `anisotropia = |Ex − Ey| / (Ex + Ey)` entre 0 y 1, y `direccion_grados` es la orientación dominante del gradiente.
- **Umbral de `apto_para_detalle`:** no se inventa. Con la columna `calidad_visual` de la etiqueta se dibuja el histograma de `nitidez_laplaciana` para "nitida" y "borrosa" y se elige el corte a ojo, anotando el valor y la figura. Hasta entonces el campo va en `null`.

Resultado: bloque `calidad` por fotograma. **Pregunta de defensa:** ¿qué le pasa al espectro de Fourier de una imagen cuando se desenfoca, y por qué?

### Etapa 3. Restauración (A, S03 + S04 + S10)

- Nuevo `restauracion.py`, que se aplica **después** de `mejorar_imagen` (gamma/CLAHE ya existentes): reducción de ruido con medias no locales (`cv2.fastNlMeansDenoisingColored`) solo si el ruido medido es alto, y máscara de enfoque (original + cantidad × (original − gaussiano)) solo si la nitidez es baja y no hay desenfoque de movimiento fuerte, porque realzar un borrón direccional crea bordes falsos.
- **El experimento que justifica la etapa.** "Mejorar la imagen para detectar mejor" es una hipótesis. `scripts/imagen_comparar_restauracion.py` corre el detector sobre los fotogramas etiquetados en tres variantes (original, mejora adaptativa actual, mejora + restauración) y cuenta, contra `etiquetas.csv`, en cuántos fotogramas se encuentra a la persona y el celular. Si la restauración no ayuda o empeora, se dice así en la exposición y la detección doble queda apagada. Ese resultado negativo también vale. El script llama directamente a `detectar_imagen` (y a `analizar_postura` si se cuenta la pose) sobre cada variante y empareja con las filas de la hoja por IoU; las personas que nadie detectó salen de la columna `personas_reales`.
- **Dónde actúa la restauración en el módulo integrado: detección doble.** `analizar_frame` ya aplica `mejorar_imagen` y detecta por dentro, y `enriquecer(original, dato)` se llama después. En lugar de cambiar ese camino, la restauración añade una **segunda pasada**: cuando la detección doble está encendida, el enriquecedor restaura el fotograma, vuelve a llamar al detector de objetos (`detectar_imagen`) sobre la imagen restaurada y guarda el resultado en `deteccion_restaurada`, comparado con la pasada normal. Así viajan las dos detecciones, antes y después de restaurar, y `analizar_frame` no se edita. La pose se calcula una sola vez, en la pasada normal.
- **Cuándo se enciende.** La constante `DETECCION_DOBLE` de `evidencia.py` nace en `False`. A la pone en `True` si en `comparacion.md` la variante restaurada encuentra más personas o más celulares reales que la mejora actual; si no, se queda apagada y el experimento se presenta como resultado. Fecha límite de la decisión: miércoles 25/11. Costo: una inferencia más del detector por fotograma, asumible con unos 10 fotogramas por clip; el tiempo se mide en la etapa 7.
- **El recorte del clasificador usa la misma variante en entrenamiento y en uso.** Si `predecir_postura` recibe el fotograma restaurado, `imagen_entrenar_postura` también debe extraer los recortes de fotogramas restaurados (o ambos del original); mezclar variantes introduce un desajuste que se confunde con error del modelo.
- Extensión opcional (solo si sobra tiempo): filtro de Wiener para borrón de movimiento con núcleo lineal estimado desde `direccion_grados`, mostrado como comparación visual.

Resultado: bloque `restauracion`, una tabla de tres columnas con el efecto medido y la decisión sobre la detección doble. **Pregunta de defensa:** ¿por qué la mediana quita el ruido sal y pimienta y el gaussiano no?

### Etapa 4. Escena (B, S04 + S05)

La cámara es fija, así que la escalera ocupa siempre los mismos píxeles. Eso permite una solución simple y fiable.

- **Calibración manual por cámara** (núcleo): `python -m scripts.imagen_calibrar_escena foto.jpg --camara <camara_id>` abre la foto; la persona hace clic en las esquinas de la zona de escalones y en los dos extremos del pasamanos. Se guarda `configs/escena/<camara_id>.json` con el polígono, el segmento y el tamaño de la imagen de referencia. Al analizar, el polígono se escala si el fotograma tiene otra resolución.
- **Evidencia por segmentación clásica** (núcleo, S05): dentro del polígono, Canny + cierre morfológico + `cv2.HoughLinesP`; se cuentan las líneas largas con orientación parecida entre sí. Cada borde físico produce al menos dos segmentos (Canny marca los dos lados de una franja clara y Hough parte una línea en trozos), así que antes de contar se **fusionan** los segmentos de orientación parecida cuya distancia perpendicular sea menor que un margen (por ejemplo 2 % del alto de la zona, mínimo 6 px); sin esa fusión el conteo sale aproximadamente al doble. `lineas_escalon_detectadas` es una medida de cuánta escalera se ve: si baja mucho respecto a la foto vacía, algo la tapa.
- **Objetos sobre escalones:** para cada objeto que no sea persona (solo existen las clases que `detector.py` deja pasar: `cell phone`, `backpack`, `handbag`, `suitcase`, `bottle`, `cup`, `chair`; ampliar la lista exigiría editar `detector.py`, que está fuera de alcance), se comprueba si el punto medio de la base de su caja cae dentro del polígono (`cv2.pointPolygonTest`).
- Extensión opcional: **resta de fondo** contra una foto de la escalera vacía (`cv2.absdiff`, umbral de Otsu, apertura morfológica, contornos) para encontrar objetos que YOLO no conoce (una caja, un charco no). Sensible a cambios de luz; se presenta como experimento.

Si no existe calibración para la cámara, `origen_zona` es `no_disponible` y los campos que dependen de ella van en `null`.

Resultado: bloque `escena` y `en_zona_escalones`. **Pregunta de defensa:** ¿por qué se aplica Canny antes de Hough, y qué hace el cierre morfológico?

### Etapa 5. Personas y objetos (B, S06)

- **Objetos en mano:** el código ya calcula `celulares[].relacion_manos`. `relaciones.py` lo generaliza a mochila, bolso, botella y vaso, usando la distancia entre el centro de la caja del objeto y cada muñeca visible, normalizada por la altura de la caja de la persona. Las muñecas se leen de `persona['pose']['puntos_corporales']` (índices 9 y 10, confianza ≥ 0,5, igual que `_conf_punto`), pero `relaciones.py` reimplementa esa lectura en lugar de importarla: importar `analisis_tiempo_real` carga YOLO (ver sección 7). Un objeto se asigna como mucho a una persona: `construir_fotograma` llama a `objetos_en_mano` para cada persona y, si un objeto queda cerca de dos, lo conserva solo en la de menor distancia relativa y pone su `detection_id` en `asociado_a_persona`.
- **Contacto con el pasamanos:** distancia de cada muñeca visible al segmento calibrado, normalizada por la altura de la persona. `si` cuando la menor distancia queda por debajo de un umbral; `no_evaluable` si no hay muñecas visibles o no hay calibración. El umbral se elige con la columna `toca_pasamanos`, igual que en la etapa 2.
- Límite que se declara: es una proyección 2D. Una mano que pasa por delante del pasamanos sin tocarlo cuenta como contacto.

Resultado: campos `objetos_en_mano`, `contacto_pasamanos`, `asociado_a_persona`. **Pregunta de defensa:** ¿por qué se divide la distancia entre la altura de la persona?

### Etapa 6. Postura con un clasificador propio (C, S06 + S11)

Es el corazón de "entrenar con nuestras escaleras", al tamaño que cabe en dos semanas: un clasificador clásico que se entiende completo, comparado con lo que ya hay.

- **Entrada:** el recorte de la persona (caja del detector), en gris, redimensionado a 64 × 128.
- **Descriptores** (`descriptores.py`): HOG (9 orientaciones, celdas de 8 × 8, bloques de 2 × 2: 3780 valores) que describe la silueta por la dirección de sus bordes, más un histograma LBP uniforme (P = 8, R = 1: 10 valores) de textura.
- **Modelo** (`clasificador_postura.py`): `StandardScaler` → `PCA` conservando 95 % de la varianza → `SVC` con kernel RBF y `class_weight='balanced'`. C y gamma se buscan con validación cruzada **agrupada por video original** (`GroupKFold` con `n_splits = min(5, número de videos originales de entrenamiento)`; `GroupKFold` falla si hay menos grupos que particiones). Las filas con `postura = no_se_ve` no entran ni en el entrenamiento ni en la evaluación; se reporta cuántas son.
- **División sin fuga:** los fotogramas de un mismo video son casi iguales, y también los de dos pedazos cortados del mismo video. Todos los de un `video_origen` van al mismo lado. Se aparta como **prueba** el 20 % de los videos originales (idealmente la segunda escalera o el segundo día completos) y no se toca hasta la etapa 7. Es el equivalente a "dividir por paciente" del caso de tuberculosis.
- **Aumento solo en entrenamiento:** volteo horizontal y cambios leves de brillo. No volteo vertical ni rotaciones grandes: cambiarían la postura.
- **Comparación obligatoria:** la regla actual por ángulo (`_evaluar_postura`) es la línea base. Se reportan las dos. `concordancia_posturas` indica en cada persona si regla y clasificador coinciden; cuando no coinciden, ese desacuerdo es información para Predicción.
- Extensión opcional: un segundo SVM cuyas características son ángulos entre puntos de la pose en lugar de píxeles. Con pocos datos puede rendir mejor; se añade solo como tercera columna de la comparación.
- El modelo entrenado se guarda en `runs/imagen/postura_svm.joblib`, carpeta ya ignorada por git; se regenera con el script. Si el archivo no existe, `postura_clasificador` vale `no_disponible` y el resto del módulo funciona igual.

Resultado: modelo entrenado y registro del entrenamiento. **Pregunta de defensa:** ¿qué es exactamente lo que cuenta un HOG, y por qué necesita menos datos que una CNN?

### Etapa 7. Evaluación (C, con A y B; S11)

`scripts/imagen_evaluar.py` compara lo que dice el módulo con `etiquetas.csv` **solo en los clips de prueba** y escribe `runs/imagen/evaluacion/informe.md` con sus figuras. Como `runs/` está en `.gitignore`, la versión final del informe y sus figuras (sin fotogramas de personas: la lista de errores lleva rutas, no imágenes) se copia a `informes/imagen_evaluacion.md` para que quede versionada y la vean los tres. La lista de clips de prueba se fija en un archivo versionado (`datos_propios/clips_prueba.txt`, excepción en `.gitignore` igual que `etiquetas.csv`) para que entrenamiento y evaluación usen la misma división.

| Campo | Métrica | Por qué |
|---|---|---|
| Postura (regla y SVM) | Matriz de confusión, recall por clase, F1 macro | Las clases están desbalanceadas: casi todo es "erguida", y la exactitud sola engaña |
| Celular en mano | Precisión y recall | Objeto pequeño; interesa saber cuántos se escapan |
| Contacto con pasamanos | Recall de "no" y exactitud | El caso que importa es quien no se agarra |
| Calidad | Separación de `nitidez_laplaciana` entre "nitida" y "borrosa" | Valida el umbral de la etapa 2 |
| Tiempo | Milisegundos por fotograma y por clip | Predicción necesita saber cuánto tarda la respuesta |

Con tan pocos clips de prueba las cifras tienen mucha incertidumbre: el informe siempre muestra los conteos (por ejemplo "11 de 14") junto al porcentaje, y una lista de errores con su fotograma para revisarlos a ojo. No hay valores meta fijados de antemano: es la primera medición que tendrá el módulo.

Resultado: informe de rendimiento con limitaciones. **Pregunta de defensa:** si casi todos los fotogramas son "erguida", ¿qué exactitud saca un modelo que siempre responde "erguida"?

### Etapa 8. Integración y demostración (A integra, 26 al 29/11; exponen los tres)

- `evidencia.py` une los bloques. `analizar_clip` recibe un parámetro opcional nuevo para que cada fotograma original pase por las piezas nuevas; sin ese parámetro se comporta exactamente como hoy. `enriquecer` escribe sus bloques bajo una clave nueva (`dato['evidencia_imagen']`) y no toca las claves existentes, para que `continuidad`, `agregar_senales` y `resumen_clip` den lo mismo que antes. Las entradas `{'estado': 'no_decodificado'}` de `resultados` no pasan por `enriquecer` y `construir_evidencia` las omite.
- Ruta nueva `POST /api/imagen/v1/evidencia` (mismo formulario que `/api/imagen/v1/clip`) que responde el JSON de la sección 4. Las rutas actuales no cambian. `clip_id` y `camara_id` salen de los metadatos validados (el archivo temporal tiene un nombre aleatorio y no sirve como `clip_id`).
- `python -m scripts.imagen_demo clip.mp4` produce el JSON y una lámina con cada fotograma en tres vistas: original, restaurado y anotado (polígono de escalones, pasamanos, cajas, esqueleto).
- **Guion de la demo (unos 8 minutos):** (1) el flujo completo en una lámina: Video recorta, Imagen describe, Predicción decide; (2) clip normal: la evidencia sale "sin novedad" y se explica cada número; (3) clip con celular y sin pasamanos: zoom al recorte, antes y después de restaurar, objeto en mano y distancia al pasamanos; (4) clip con paso rápido: el borrón se mide y se reporta en vez de tirarse; (5) clip agachado u objeto en el escalón: regla y clasificador, coincidan o no; (6) la tabla de la etapa 7 y dos errores reales con su explicación; (7) limitaciones y qué falta. Llevar los resultados **pregrabados** por si falla la máquina.

Resultado: módulo integrado, demo ensayada y cada integrante con su etapa.

## 6. Calendario

| Días | A | B | C |
|---|---|---|---|
| Lun 16 – Mar 17/11 | Contrato y ejemplo a Predicción | Grabar clips; foto de escalera vacía | Guion de grabación; grabar |
| Mié 18 – Dom 22/11 | Etapas 2 y 3 sobre los fotogramas | Calibración y etapa 4 | Extraer, etiquetar (con ayuda), descriptores |
| Lun 23 – Mié 25/11 | Experimento de restauración | Etapa 5 | Entrenar y comparar con la regla |
| Jue 26 – Dom 29/11 | Integración, ruta y demo | Umbrales con etiquetas; revisar errores | Evaluación e informe |
| Semana 14 | Ensayo general y defensa | | |

Punto de control el domingo 22/11: cada pieza corre sola sobre un clip y produce su bloque. Lo que no esté, se recorta a su versión mínima (ver "Si son dos personas").

## 7. Cambios en el código

### Archivos nuevos

| Ruta | Contenido |
|---|---|
| `app/modulo_imagen/nitidez.py` | `energia_alta_frecuencia`, `desenfoque_movimiento`, `bloque_calidad` |
| `app/modulo_imagen/restauracion.py` | `restaurar` |
| `app/modulo_imagen/escena.py` | `cargar_calibracion`, `escalar_calibracion`, `contar_lineas_escalon`, `punto_en_zona`, `bloque_escena` |
| `app/modulo_imagen/relaciones.py` | `objetos_en_mano`, `contacto_pasamanos` |
| `app/modulo_imagen/descriptores.py` | `recorte_persona`, `vector_hog_lbp` |
| `app/modulo_imagen/clasificador_postura.py` | `crear_pipeline`, `entrenar`, `cargar_modelo`, `predecir_postura` |
| `app/modulo_imagen/evidencia.py` | `construir_fotograma`, `construir_evidencia`, `validar_evidencia`, `evidencia_de_clip` |
| `scripts/imagen_preparar_etiquetado.py` | Extrae fotogramas y crea la plantilla `etiquetas.csv` |
| `scripts/imagen_calibrar_escena.py` | Clics para polígono y pasamanos |
| `scripts/imagen_comparar_restauracion.py` | Experimento de la etapa 3 |
| `scripts/imagen_entrenar_postura.py` | Entrena y guarda el SVM |
| `scripts/imagen_evaluar.py` | Informe de la etapa 7 |
| `scripts/imagen_demo.py` | JSON y lámina para la demo |
| `configs/escena/.gitkeep` | Carpeta de calibraciones |
| `requirements-imagen.txt` | Dependencias mínimas del módulo, sin los paquetes CUDA, todas con versión fijada |
| `instalar_imagen.ps1` | Instalador para Windows: crea el entorno, instala y corre las pruebas |
| `tests/fixtures/evidencia_ejemplo.json` | El ejemplo de la sección 4 |
| `tests/test_nitidez.py`, `test_restauracion.py`, `test_escena.py`, `test_relaciones.py`, `test_descriptores.py`, `test_clasificador_postura.py`, `test_evidencia.py` | Pruebas con imágenes sintéticas, sin cargar YOLO |
| `tests/test_clip_enriquecer.py`, `tests/test_http_evidencia.py` | Parámetro `enriquecer` de `analizar_clip` y ruta nueva, con inferencias simuladas por `monkeypatch` (mismo patrón que `test_clip_seguimiento_etapa4.py` y `test_http_integracion_video_imagen.py`) |
| `informes/imagen_evaluacion.md` | Copia versionada del informe final de la etapa 7 |

**Regla de importaciones (evita cargar YOLO en las pruebas).** `detector.py` y `pose.py` crean `YOLO("yolo11n.pt")` y `YOLO("yolo11n-pose.pt")` al importarse, y `analisis_tiempo_real.py` y `analizador_clip.py` los importan. Por eso ningún archivo nuevo de `app/modulo_imagen/` importa esos cuatro módulos en la cabecera: `evidencia_de_clip` importa `analizar_clip` dentro de la función, y `relaciones.py` lee las muñecas por su cuenta. Así `test_evidencia.py` y los demás corren sin pesos ni descargas.

### Archivos que se modifican

| Ruta | Cambio |
|---|---|
| `app/modulo_imagen/analizador_clip.py` | `analizar_clip` gana el parámetro `enriquecer=None`. Si se pasa, se llama `enriquecer(original, dato)` justo antes de `agregar_senales(dato)`. Nada más cambia. |
| `app/integracion_imagen.py` | Ruta nueva `POST /api/imagen/v1/evidencia`. Hoy las validaciones de `recibir_clip` (tamaño, `_configuracion('clip')`, `intervalo_s` entre 0,1 y 5,0, extensión) están escritas dentro de la propia función, así que primero se extraen a un auxiliar `_leer_peticion_clip()` que devuelve `(contenido, metadatos, modo, intervalo, extension)` y lanza las mismas excepciones; `recibir_clip` pasa a usarlo sin cambiar sus respuestas (lo comprueban las pruebas HTTP existentes). La ruta nueva importa `evidencia_de_clip` dentro de la función (import diferido, como hace `/clip`) y borra el temporal en `finally`. |
| `.gitignore` | Ya ignora `*.mp4` y `runs/`. Añadir `datos_propios/*` con las excepciones `!datos_propios/etiquetas.csv` y `!datos_propios/clips_prueba.txt`, para que tampoco se suban fotogramas sueltos ni la foto de la escalera vacía. |
| `tests/test_http_integracion_video_imagen.py` | No se modifica; la ruta nueva se prueba en un archivo aparte (`tests/test_http_evidencia.py`). |
| `MODULO_IMAGEN_PASO1.md` | Un párrafo que remite a este plan y al contrato. |

### Firmas

```python
# nitidez.py
def energia_alta_frecuencia(frame: np.ndarray, radio_rel: float = 0.10) -> float: ...   # 0..1
def desenfoque_movimiento(frame: np.ndarray) -> dict: ...   # {'anisotropia': float, 'direccion_grados': float}
def bloque_calidad(frame: np.ndarray, *, umbral_nitidez: float | None = None) -> dict: ...

# restauracion.py
def restaurar(frame: np.ndarray, calidad: dict, *, modo: str = 'adaptativo') -> tuple[np.ndarray, dict]: ...   # modo: 'ninguno' | 'adaptativo'

# escena.py
def cargar_calibracion(camara_id: str, carpeta: Path = Path('configs/escena')) -> dict | None: ...
def escalar_calibracion(calibracion: dict, ancho: int, alto: int) -> dict: ...
def contar_lineas_escalon(frame: np.ndarray, zona: list[list[int]]) -> int: ...
def punto_en_zona(punto: tuple[float, float], zona: list[list[int]]) -> bool: ...
def bloque_escena(frame: np.ndarray, calibracion: dict | None) -> dict: ...

# relaciones.py
def objetos_en_mano(persona: dict, objetos: list[dict], *, umbral_rel: float = 0.18) -> list[dict]: ...
def contacto_pasamanos(persona: dict, segmento: list[int] | None, *, umbral_rel: float = 0.12) -> str: ...

# descriptores.py
def recorte_persona(frame: np.ndarray, bbox: list[float], tam: tuple[int, int] = (64, 128)) -> np.ndarray | None: ...   # gris uint8 de 128x64, o None si la caja es degenerada
def vector_hog_lbp(recorte: np.ndarray) -> np.ndarray: ...   # float32, longitud 3790

# clasificador_postura.py
CLASES = ('erguida', 'inclinada', 'agachada_o_sentada', 'horizontal')
def crear_pipeline(C: float = 1.0, gamma: str | float = 'scale') -> Pipeline: ...
def entrenar(X: np.ndarray, y: np.ndarray, grupos: np.ndarray) -> tuple[Pipeline, dict]: ...
def cargar_modelo(ruta: Path = Path('runs/imagen/postura_svm.joblib')) -> Pipeline | None: ...
def predecir_postura(frame: np.ndarray, bbox: list[float], modelo: Pipeline | None) -> str: ...

# evidencia.py
VERSION = 'imagen-evidencia-1.0'
DETECCION_DOBLE = False   # A la cambia a True si el experimento de la etapa 3 muestra mejora
CLAVES_PROHIBIDAS = ('riesgo', 'probabilidad', 'alerta', 'accidente')
def construir_fotograma(dato: dict) -> dict: ...
def construir_evidencia(resultado_clip: dict, *, clip_id: str, camara_id: str) -> dict: ...   # resultado_clip = primer elemento que devuelve analizar_clip
def validar_evidencia(evidencia: dict) -> None: ...   # ValueError si falta un campo obligatorio o si alguna clave, a cualquier profundidad y en minúsculas, contiene una palabra de CLAVES_PROHIBIDAS
def evidencia_de_clip(ruta_clip, *, clip_id: str | None = None, camara_id: str = 'camara_local', contexto_video: dict | None = None, modo: str = 'adaptativo', intervalo_s: float = 1.0, modelo_postura=None, calibracion: dict | None = None, deteccion_doble: bool | None = None) -> dict: ...   # clip_id por defecto: Path(ruta_clip).stem
```

`construir_fotograma` toma los objetos de `dato['celulares']` (que ya incluye los de la segunda pasada `roi_mano_exploratorio`) más `dato['otros_objetos']`, y las personas según la regla de la sección 4. La comprobación de claves prohibidas es por subcadena, así que antes de añadir un campo nuevo se comprueba que no contenga ninguna de esas palabras.

Los valores por defecto `umbral_rel` de `relaciones.py` son provisionales y se sustituyen por los elegidos con las etiquetas (etapa 5).

Correspondencia de nombres al construir el fotograma: `estado_postural` del código actual pasa a `postura_reglas_pose` sin cambiar sus valores; `confianza` pasa a `confianza_detector`; `pose.inclinacion_torso` pasa a `inclinacion_torso_grados`.

### Dependencias nuevas

`scikit-learn==1.9.1`, `scikit-image==0.26.0` y `joblib==1.6.0`, compatibles con `numpy==1.26.4` y Python 3.11, que es lo que fija el proyecto. Van en `requirements-imagen.txt` junto con `numpy==1.26.4`, `opencv-python==4.10.0.84`, `ultralytics==8.4.170`, `matplotlib==3.11.2`, `Flask==3.1.3`, `Werkzeug==3.1.5` (la ruta nueva y las pruebas HTTP los necesitan; sin Flask esas pruebas se saltan en silencio por `importorskip`) y `pytest`. La primera instalación en Windows confirma que las tres tienen rueda para Python 3.11; si alguna no instala, se baja a la versión anterior más cercana y se anota aquí. `pytest` también se fija a la versión que quede instalada ese día. `ultralytics` arrastra `torch`: en Windows el `torch` de PyPI es solo CPU, pero en Linux trae los paquetes `nvidia-*`; quien trabaje en Linux instala antes `torch` y `torchvision` desde `https://download.pytorch.org/whl/cpu` (se anota como comentario en el archivo). `requirements.txt` general no se toca.

**Instalador.** `instalar_imagen.ps1`, en la raíz de `Video-G-G-dev/`, hace tres cosas: crea el entorno `.venv_imagen` con `py -3.11 -m venv` si no existe, instala `requirements-imagen.txt` y corre `python -m pytest tests -q`. Termina con un mensaje claro de "entorno listo" o con el error. Es el mismo script para las pruebas de ahora, para la integración con Video y para la máquina de la demo: lo que cambia con el tiempo es la lista de paquetes, no el script. Regla de equipo: **ningún paquete nuevo entra sin su línea, con versión, en `requirements-imagen.txt`.** La máquina que además corra el servidor de Video instala las dependencias de Video como indica su README (los paquetes CUDA dependen de la GPU y no se instalan a ciegas).

## 8. Qué no se toca

- `app/risk_engine.py`, `app/server.py`, `scripts/train.py` y todo el entrenamiento del detector de Video.
- Las rutas `/api/imagen/v1/salud`, `/fotograma` y `/clip`, y la forma de sus respuestas.
- `calidad.py`, `preprocesamiento.py`, `detector.py`, `pose.py`, `analisis_tiempo_real.py`, `senalizacion.py`, `continuidad_visual.py`, `contrato_video.py`: se reutilizan sin editar (la detección doble llama a `detectar_imagen` una segunda vez desde `evidencia.py`; no cambia `analizar_frame`). Las claves que ya devuelven `medir_calidad` y `analizar_frame` no cambian.
- Las pruebas existentes en `tests/`: deben seguir pasando sin modificarlas.
- Fuera de alcance: porcentajes o niveles de riesgo, alertas, seguimiento de identidades entre fotogramas, entrenar una CNN o reentrenar YOLO, detectar piso mojado.

## 9. Casos límite

| Situación | Comportamiento esperado |
|---|---|
| Fotograma sin personas | `personas: []`; `calidad` y `escena` se calculan igual |
| Fotograma muy borroso | Se analiza igual; `apto_para_detalle: false`; no se aplica máscara de enfoque si la anisotropía es alta |
| Fotograma que no decodifica | No aparece en `fotogramas`; `resumen.fotogramas_analizados` lo refleja |
| No hay calibración para la cámara | Se añade a `limitaciones` la frase "Sin calibración para la cámara '<camara_id>'", para que el fallo no pase en silencio; `origen_zona: "no_disponible"`; `zona_escalones`, `en_zona_escalones` en `null`; `contacto_pasamanos: "no_evaluable"` |
| Resolución distinta a la de la calibración | El polígono y el segmento se escalan proporcionalmente |
| Muñecas no visibles | `objetos_en_mano: []`, `contacto_pasamanos: "no_evaluable"` |
| Caja de persona degenerada o fuera de la imagen | `recorte_persona` devuelve `None`; `postura_clasificador: "no_evaluable"` |
| No existe `postura_svm.joblib` | `postura_clasificador: "no_disponible"`, `concordancia_posturas: null` |
| Persona tapada a medias | Se clasifica el recorte disponible; es un límite declarado en `limitaciones` |
| Imagen totalmente uniforme | `energia_alta_frecuencia` y `anisotropia` devuelven 0.0 sin dividir entre cero |
| Aparece una clave prohibida en la salida | `validar_evidencia` lanza `ValueError` |
| Detección doble apagada | `deteccion_restaurada: null` |
| Detección doble encendida y la restauración no aplica ninguna operación | No se repite la detección; `deteccion_restaurada` con los mismos conteos de la pasada normal y `solo_tras_restaurar: []` |
| Dos personas en el fotograma | Una entrada por persona; cada objeto se asocia como mucho a la persona más cercana |
| Pose sin caja YOLO o persona recuperada por ROI | Entra en `personas` con su `origen`, `detection_id: null` y, si no hay caja YOLO, `confianza_detector: null` |
| Video manda `riesgo_reportado_por_video` en `contexto_video` | No se copia a la evidencia; `validar_evidencia` no falla |
| Clasificador dice `agachada_o_sentada` | `concordancia_posturas: null` (la regla no tiene esa clase) |

## 10. Pruebas

Todas con imágenes sintéticas hechas con NumPy y OpenCV, sin cargar modelos, al estilo de `tests/test_adaptativo.py`.

| Archivo | Escenario | Esperado |
|---|---|---|
| `test_nitidez.py` | Tablero de ajedrez nítido frente al mismo con `GaussianBlur` | `energia_alta_frecuencia` del nítido es mayor |
| | Tablero con borrón horizontal (núcleo 1 × 15) | `anisotropia` mayor que en el original |
| | Imagen uniforme | 0.0 en ambas medidas, sin error |
| `test_restauracion.py` | Imagen con ruido añadido y `ruido_mad_estimado` alto | Menor diferencia media respecto a la limpia; la entrada no se muta |
| | `modo='ninguno'` | Salida idéntica y `operaciones == []` |
| `test_escena.py` | Imagen negra con 6 líneas blancas horizontales dentro del polígono | `contar_lineas_escalon` entre 5 y 7 |
| | Punto dentro y fuera del polígono | `True` y `False` |
| | Calibración de 1280 × 720 aplicada a 640 × 360 | Coordenadas a la mitad |
| | `bloque_escena(frame, None)` | `origen_zona == 'no_disponible'` |
| `test_relaciones.py` | Muñeca a 5 px del segmento, persona de 300 px de alto | `'si'` |
| | Muñeca a 150 px | `'no'` |
| | Sin muñecas visibles o sin segmento | `'no_evaluable'` |
| | Celular centrado en la muñeca derecha | Un elemento con `mano == 'derecha'` |
| `test_descriptores.py` | Recorte válido | Forma (128, 64), `uint8`; vector de longitud 3790 y finito |
| | Caja de ancho cero | `None` |
| `test_clasificador_postura.py` | Datos sintéticos: rectángulo vertical frente a horizontal, 40 muestras en 8 grupos | El pipeline entrena y acierta más de 90 % en entrenamiento |
| | `cargar_modelo` con ruta inexistente | `None`; `predecir_postura(..., None) == 'no_disponible'` |
| `test_evidencia.py` | `tests/fixtures/evidencia_ejemplo.json` | `validar_evidencia` no lanza |
| | Ejemplo con una clave `riesgo` añadida | `ValueError` |
| | `construir_fotograma` con un `dato` mínimo como el de `analizar_frame` | Renombra `estado_postural` y `confianza`; bloques ausentes en `null` |
| | `dato` con una entrada en `poses_sin_asociar` | Aparece en `personas` con `origen == 'solo_pose'` y `detection_id is None` |
| | `construir_evidencia` con `contexto_video` que trae `riesgo_reportado_por_video: 'ALTO'` | La clave no aparece en la salida y `validar_evidencia` no lanza |
| | Resultado de clip con una entrada `{'estado': 'no_decodificado'}` | No aparece en `fotogramas` |
| | Un celular cerca de las muñecas de dos personas | Solo figura en `objetos_en_mano` de la más cercana |
| | Regla `inclinacion_para_revision` con clasificador `inclinada`; y clasificador `agachada_o_sentada` | `concordancia_posturas` `True`; y `None` |
| `test_escena.py` (adicional) | Una sola franja blanca de 4 px de grosor | `contar_lineas_escalon == 1` (los dos bordes se fusionan) |
| `test_clip_enriquecer.py` | Clip sintético `.avi` con `analizar_frame`, `recuperar_en_roi` y `recuperar_pose_en_roi` simulados; `enriquecer` que añade `dato['evidencia_imagen']` | Se llama una vez por fotograma decodificado con el fotograma original; sin `enriquecer`, el resultado es igual que antes (mismas claves y `resumen_visual`) |
| `test_http_evidencia.py` | `evidencia_de_clip` simulado por `monkeypatch` en `sys.modules`; POST con `clip_id` en metadatos | 200, `schema_version == 'imagen-evidencia-1.0'`, `clip.clip_id` igual al de los metadatos, temporal borrado; extensión `.txt` o `intervalo_s=9` → 400 |

## 11. Verificación

Desde `Video-G-G-dev/`, con el entorno virtual activo:

```
.\instalar_imagen.ps1
python -m pytest tests -q
python -m scripts.imagen_preparar_etiquetado datos_propios/videos --fps 2
python -m scripts.imagen_calibrar_escena datos_propios/escalera_vacia.jpg --camara escalera_g
python -m scripts.imagen_comparar_restauracion --etiquetas datos_propios/etiquetas.csv
python -m scripts.imagen_entrenar_postura --etiquetas datos_propios/etiquetas.csv
python -m scripts.imagen_evaluar --etiquetas datos_propios/etiquetas.csv
python -m scripts.imagen_demo datos_propios/clips/g2_1117_a1_celular_02.mp4 --camara escalera_g
```

Hecho cuando: todas las pruebas pasan, incluidas las anteriores; `imagen_demo` produce un JSON que `validar_evidencia` acepta; existe `runs/imagen/evaluacion/informe.md` con matriz de confusión sobre clips de prueba y su copia versionada en `informes/imagen_evaluacion.md`; Predicción confirma que lee el JSON; y cada integrante contesta la pregunta de defensa de su etapa.

## 12. Riesgos y cómo se acotan

| Riesgo | Qué se hace |
|---|---|
| Dos semanas no alcanzan | Núcleo y extensiones separados; punto de control el 22/11 |
| El SVM rinde poco con pocos datos | Se reporta junto a la regla de pose; un resultado modesto bien medido sigue siendo defendible. Más clips de la clase que falla antes que más técnica |
| Predicción no responde a tiempo | El contrato y el ejemplo se envían igual el 17/11; añadir campos no rompe nada |
| La restauración no mejora la detección | Se presenta el experimento y la detección doble queda apagada |
| Pocos videos originales distintos | Cortar videos largos aumenta fotogramas, no variedad: se graban clips cortos de las posturas que falten y se reporta cuántos videos originales hay por clase |
| `camara_id` distinto entre Video y la calibración | Texto exacto pactado en la etapa 0; aviso en `limitaciones` cuando no hay calibración |
| Etiquetas inconsistentes | Doble etiquetado de 40 filas en la etapa 1 |
| Privacidad | Alcance académico: códigos de actor e imágenes fuera de git; terceros en la grabación se declaran como limitación |

## 13. Cómo se adapta el caso de tuberculosis a este

| Elemento | Caso de tuberculosis | Módulo Imagen en escaleras |
|---|---|---|
| Entrada | Una radiografía | Un clip de riesgo, visto como fotogramas |
| Salida | Probabilidad y etiqueta | Evidencia descriptiva; sin probabilidad de accidente |
| Quién decide | El radiólogo | El subsistema de Predicción |
| Unidad que no se puede partir | El paciente | El clip |
| Preprocesamiento | Gris, tamaño, CLAHE | Medir calidad, gamma/CLAHE, restauración condicionada |
| Región de interés | Pulmones segmentados | Zona de escalones calibrada y recorte de cada persona |
| Características | LBP/HOG o CNN | HOG + LBP del recorte; ángulos de la pose |
| Modelo | SVM de línea base frente a CNN | SVM propio frente a la regla de pose ya existente |
| Métrica prioritaria | Sensibilidad | Recall por clase de postura y recall de "no toca pasamanos" |
| Error más costoso | Falso negativo | No reportar una postura anómala o un objeto en el escalón |

## Due diligence record

Esta sección la escribe la revisión de diligencia debida (14/11/2026) y la leen las pasadas posteriores, incluida la comprobación de hechos durante la implementación. Registra lo que se concluyó una vez al verificar los hechos externos del plan, no lo que es cierto ahora: ninguna pasada puede usar una línea de esta sección como motivo para saltarse una verificación. Si una persona cambia a mano alguno de estos hechos en el plan, debe **borrar** la línea correspondiente en lugar de actualizarla, para que la sección no afirme cosas que nadie comprobó.

- SETTLED: `ultralytics==8.4.170` (verificado en pypi.org/project/ultralytics/8.4.170: existe, publicado el 30/09/2026, Requires-Python >=3.8)
- SETTLED: `matplotlib==3.11.2` (verificado en pypi.org/project/matplotlib/3.11.2: existe, Requires-Python >=3.11, hay rueda `cp311-win_amd64`)
- SETTLED: `Flask==3.1.3` (verificado en pypi.org/project/Flask/3.1.3: existe, Requires-Python >=3.9)
- SETTLED: `Werkzeug==3.1.5` (verificado en pypi.org/project/Werkzeug/3.1.5: existe, Requires-Python >=3.9)
- SETTLED: `opencv-python==4.10.0.84` (verificado en pypi.org/project/opencv-python/4.10.0.84: existe, rueda `cp37-abi3-win_amd64` instalable en Python 3.11)
- SETTLED: `https://download.pytorch.org/whl/cpu` (verificado en el propio índice: existe y sirve ruedas `+cpu` de torch para cp311 en Linux y Windows)
- SETTLED: `pip install --dry-run` (verificado en pip.pypa.io, documentación de `pip install`)
- SETTLED: HOG de 64 × 128 con 9 orientaciones, celdas 8 × 8 y bloques 2 × 2 da 3780 valores, y LBP `uniform` con P = 8, R = 1 da 10 códigos, total 3790 (comprobado ejecutando scikit-image 0.26.0; ojo: el valor por defecto de `hog` es `cells_per_block=(3, 3)`, que da 6804, así que `(2, 2)` hay que pasarlo explícitamente)
- SETTLED: `GroupKFold` exige `n_splits >= 2` y no más particiones que grupos; si no, lanza `ValueError` (documentación de scikit-learn y comprobado con scikit-learn 1.9.1)
- SETTLED: puntos COCO 9 = muñeca izquierda y 10 = muñeca derecha, y los valores de `_evaluar_postura` (comprobado en `app/modulo_imagen/analisis_tiempo_real.py`)
- SETTLED: 31/08/2026, 16/11/2026 y 30/11/2026 caen en lunes, coherente con semana 12 = 16/11 y semana 14 = 30/11 al 04/12 (aritmética de calendario; el sílabo en sí no se consultó)
- OPEN (unverified): que el `torch` de PyPI en Windows sea solo CPU y en Linux arrastre los paquetes `nvidia-*`; la página de PyTorch no lo afirma de forma explícita
- OPEN (unverified): que exista rueda `cp311-win_amd64` de `scikit-image==0.26.0`; el listado de PyPI salió truncado (lo resuelve el `--dry-run` en Windows)
- OPEN (currency): `ultralytics==8.4.170` es válido, pero PyPI muestra también 8.4.171; se deja el pin del proyecto
- OPEN (deferred): qué campos usará Predicción y si Video envía `clip_id` y `camara_id`; se cierra en la reunión de la etapa 0

## Implementation steps

Comandos de verificación del proyecto (no hay `CLAUDE.md` ni `pytest.ini`; se ejecutan desde `Video-G-G-dev/` con el entorno virtual activo): `python -m pip install -r requirements-imagen.txt` para dependencias, `python -m pytest tests -q` para toda la batería y `python -m pytest tests/<archivo>.py -q` para un archivo. No hay linter ni comprobador de tipos configurado. Los scripts se ejecutan como `python -m scripts.<nombre>` (la carpeta `scripts/` funciona como paquete de espacio de nombres) y cada uno debe responder a `--help` sin cargar YOLO (importaciones de `detector`, `pose`, `analisis_tiempo_real`, `analizador` y `analizador_clip` siempre dentro de funciones, regla de la sección 7). Ninguna prueba existente de `tests/` se modifica y todas deben seguir pasando en cada paso.

Orden y responsables: el paso 1 (A) y el paso 2 (C) van primero; los pasos 3 (A), 4 (B) y 5 (C) son independientes entre sí y se pueden hacer en paralelo; el paso 6 (A) une las piezas; el paso 7 (A) y el paso 8 (C) dependen solo del paso 6. Decisión cerrada sobre la restauración: detección doble opcional desde `evidencia.py` (constante `DETECCION_DOBLE`, apagada por defecto), sin editar `analizar_frame`; A la enciende tras el experimento del paso 3 si muestra mejora.

### Paso 1: Contrato `imagen-evidencia-1.0` y base del módulo (A, etapa 0)

- **Files**: crear `requirements-imagen.txt`, `instalar_imagen.ps1`, `configs/escena/.gitkeep`, `tests/fixtures/evidencia_ejemplo.json`, `tests/test_evidencia.py`, `app/modulo_imagen/evidencia.py`; modificar `.gitignore`.
- **Do**:
  - `tests/fixtures/evidencia_ejemplo.json`: copia literal del JSON de la sección 4 (debe pasar la validación tal cual).
  - `tests/test_evidencia.py`: las pruebas de la lista **Test** de abajo, construyendo `dato` mínimos a mano con la forma que devuelve `analizar_frame` (claves `personas`, `poses_sin_asociar`, `celulares`, `otros_objetos`, `calidad_original`, `segundo_s`, `fotograma_indice`); sin importar YOLO.
  - `app/modulo_imagen/evidencia.py` con `VERSION = 'imagen-evidencia-1.0'` y `CLAVES_PROHIBIDAS = ('riesgo', 'probabilidad', 'alerta', 'accidente')`. Cabecera sin importar `detector`, `pose`, `analisis_tiempo_real` ni `analizador_clip`.
  - `validar_evidencia(evidencia) -> None`: lanza `ValueError` si falta `schema_version` (o no es igual a `VERSION`), `clip.clip_id`, la lista `fotogramas`, en cada fotograma `segundo_s` y `calidad` (dict con `nitidez_laplaciana`) y la lista `personas`, en cada persona `bbox`, `postura_reglas_pose` y `objetos_en_mano` (lista); también lanza si una clave, a cualquier profundidad (dicts dentro de listas incluidos) y en minúsculas, contiene por subcadena alguna palabra de `CLAVES_PROHIBIDAS`. Comprueba además que los campos con valores cerrados de la sección 4 (`postura_clasificador`, `contacto_pasamanos`, `origen_zona`, `mano`, `postura_reglas_pose`, `origen`) tengan un valor permitido o `null`.
  - Auxiliar público `personas_de_dato(dato) -> list[dict]`: devuelve, en este orden fijo, las personas de `dato['personas']` (`origen: 'yolo'`, con su `detection_id`, `bbox`, `confianza`, `pose`, `estado_postural`), cada elemento de `dato['poses_sin_asociar']` envuelto como `{'origen': 'solo_pose', 'detection_id': None, 'bbox': pose['bbox'], 'confianza': None, 'pose': pose, 'estado_postural': pose['estado_postural']}`, y la persona de `dato['recuperacion_roi']` si tiene `origen` (`roi_yolo` con `confianza` y sin pose; `roi_pose` con la pose y `confianza: None`; leer `recuperacion_roi.py`, `pose_roi.py` y `analizador_clip.py` para la forma exacta). Para `roi_yolo` sin `estado_postural`, usar `no_evaluable`. El paso 6 usa este mismo orden para alinear posturas del clasificador.
  - Forma interna acordada de `dato['evidencia_imagen']` (la rellenará el paso 6; aquí solo se lee si existe): `{'calidad': dict | None, 'restauracion': dict | None, 'deteccion_restaurada': dict | None, 'escena': dict | None, 'calibracion': dict | None, 'posturas_clasificador': list[str] | None}`, donde `posturas_clasificador` va alineada con `personas_de_dato(dato)`.
  - `construir_fotograma(dato) -> dict`: devuelve `indice` (`dato.get('fotograma_indice')`), `segundo_s`, `calidad` (el bloque de `evidencia_imagen` si existe; si no, `luminosidad`, `contraste`, `nitidez_laplaciana` y `ruido_mad_estimado` tomados de `dato['calidad_original']` con `energia_alta_frecuencia`, `desenfoque_movimiento` y `apto_para_detalle` en `null`), `restauracion`, `deteccion_restaurada` y `escena` (de `evidencia_imagen` o `null`), `personas` y `objetos`. Por persona: `detection_id`, `origen`, `bbox`, `confianza_detector` (renombra `confianza`), `inclinacion_torso_grados` (de `pose.inclinacion_torso`, o `null`), `postura_reglas_pose` (renombra `estado_postural` sin cambiar el valor), `postura_clasificador` (de `posturas_clasificador[i]` o `'no_disponible'`), `concordancia_posturas`, y de momento `en_zona_escalones: null`, `contacto_pasamanos: 'no_evaluable'`, `objetos_en_mano: []` (el paso 6 los calcula). Objetos: `dato['celulares']` + `dato['otros_objetos']` con `clase`, `bbox`, `confianza_detector`, `en_zona_escalones: null`, `asociado_a_persona: null`; no se copian `relacion_manos` ni ninguna otra clave del código actual.
  - Correspondencia para `concordancia_posturas` (constante de módulo): `sin_indicio_suficiente` ↔ `erguida`, `inclinacion_para_revision` ↔ `inclinada`, `postura_horizontal_posible` ↔ `horizontal`; `True`/`False` si ambos lados tienen equivalente, `None` si el clasificador dice `agachada_o_sentada`, `no_evaluable` o `no_disponible`, o si la regla dice `no_evaluable`.
  - `construir_evidencia(resultado_clip, *, clip_id, camara_id) -> dict`: `schema_version`; `clip` con `clip_id`, `camara_id`, `duracion_s`, `fps_origen` y `contexto_video` con **solo** `escaleras` y `tracks` (lista vacía si faltan); `fotogramas` con `construir_fotograma` de cada entrada de `resultado_clip['resultados']` cuyo `estado` no sea `no_decodificado`; `resumen` (`fotogramas_analizados` = fotogramas incluidos, `personas_max`, `fotogramas_con_desenfoque` = fotogramas con `apto_para_detalle is False`, `posturas_observadas` = conteo de `postura_clasificador` sin `no_disponible`/`no_evaluable`, `clases_objeto` = conteo de `objetos[].clase`); `limitaciones` con al menos el límite de proyección 2D y que es evidencia sin interpretación. No copia `resumen_visual`, `senales_visuales` ni `nota`.
  - `requirements-imagen.txt` con los pines de la sección 7 (`numpy==1.26.4`, `opencv-python==4.10.0.84`, `ultralytics==8.4.170`, `matplotlib==3.11.2`, `Flask==3.1.3`, `Werkzeug==3.1.5`, `pytest`) más `scikit-learn==1.9.1`, `scikit-image==0.26.0` y `joblib==1.6.0` (si alguna no instala en Windows con Python 3.11, bajar a la versión anterior más cercana y actualizar la sección 7); fijar también `pytest` a la versión instalada; comentario sobre instalar `torch`/`torchvision` CPU desde `https://download.pytorch.org/whl/cpu` en Linux. `requirements.txt` no se toca.
  - `instalar_imagen.ps1` en la raíz de `Video-G-G-dev/`: con `$ErrorActionPreference = 'Stop'`, crea `.venv_imagen` con `py -3.11 -m venv .venv_imagen` si no existe, ejecuta `.\.venv_imagen\Scripts\python.exe -m pip install -r requirements-imagen.txt` y luego `.\.venv_imagen\Scripts\python.exe -m pytest tests -q`; comprueba `$LASTEXITCODE` tras cada comando y termina con "Entorno de Imagen listo" o con el paso que falló. No instala `requirements.txt` general. Se puede volver a ejecutar sin romper nada.
  - `.gitignore`: añadir `.venv_imagen/`, `datos_propios/*`, `!datos_propios/etiquetas.csv` y `!datos_propios/clips_prueba.txt`.
- **Test** (`tests/test_evidencia.py`): el fixture pasa `validar_evidencia`; el fixture con una clave `riesgo` añadida dentro de una persona lanza `ValueError`; un `contacto_pasamanos: 'quizas'` lanza `ValueError`; `construir_fotograma` con un `dato` mínimo (una persona YOLO con pose) renombra `estado_postural` → `postura_reglas_pose` y `confianza` → `confianza_detector`, y deja `restauracion` y `escena` en `None` con `calidad.nitidez_laplaciana` tomada de `calidad_original`; un `dato` con una entrada en `poses_sin_asociar` aparece en `personas` con `origen == 'solo_pose'` y `detection_id is None`; un `dato` con `recuperacion_roi` de `origen: 'roi_yolo'` aparece con ese origen; el fixture con `deteccion_restaurada: null` también valida; `construir_evidencia` con `contexto_video` que trae `riesgo_reportado_por_video: 'ALTO'` no copia la clave y la salida valida; una entrada `{'segundo_s': 2.0, 'estado': 'no_decodificado'}` no aparece en `fotogramas` y `resumen.fotogramas_analizados` cuenta solo las decodificadas; con `evidencia_imagen.posturas_clasificador = ['inclinada']` y regla `inclinacion_para_revision`, `concordancia_posturas is True`, y con `['agachada_o_sentada']` es `None`.
- **Verify**: en Windows, `.\instalar_imagen.ps1` termina con "Entorno de Imagen listo" (en otro sistema, `python -m pip install -r requirements-imagen.txt` sin errores); `python -m pytest tests/test_evidencia.py -q` pasa (unas 10 pruebas); `python -m pytest tests -q` pasa completo. Enviar `tests/fixtures/evidencia_ejemplo.json` a Predicción.

### Paso 2: Extracción de fotogramas y plantilla de etiquetado (C, etapa 1)

- **Files**: crear `scripts/imagen_preparar_etiquetado.py`.
- **Do**:
  - Funciones auxiliares sin YOLO, importables por los scripts de los pasos 3, 5 y 8: `COLUMNAS` (lista fija de columnas), `iou(a, b) -> float` (cajas `[x1, y1, x2, y2]`, 0.0 si no hay unión), `leer_etiquetas(ruta) -> list[dict]` (convierte `segundo_s` a float, `bbox` a lista de float, `detection_id`, `personas_reales` y `celulares_reales` a int o `None` si están vacíos) y `emparejar_por_iou(bbox, filas, umbral=0.5) -> dict | None` (la fila de tipo persona con mayor IoU ≥ umbral).
  - Columnas de `etiquetas.csv`, en este orden: `clip_id`, `video_origen`, `fotograma`, `segundo_s`, `tipo_fila` (`fotograma` o `persona`), `detection_id`, `origen`, `bbox` (texto JSON `[x1, y1, x2, y2]`), `personas_reales`, `celulares_reales`, `postura`, `objeto_en_mano`, `toca_pasamanos`, `calidad_visual`, `notas`. La columna `tipo_fila` es necesaria porque las personas `solo_pose` también tienen `detection_id` vacío y no deben confundirse con la fila de fotograma.
  - `main()` con `argparse`: argumento posicional carpeta de videos (por defecto `datos_propios/videos`), `--segmento-s` (por defecto 10), `--fps` (por defecto 2), `--salida` (por defecto `datos_propios`), `--modo` (por defecto `adaptativo`), `--sobrescribir`. Primero prepara los clips en `<salida>/clips/`: un video de hasta `--segmento-s` segundos se copia tal cual (`clip_id` = `video_origen` = nombre sin extensión); uno más largo se corta con `cv2.VideoCapture` y `cv2.VideoWriter` (`mp4v`, mismos FPS y tamaño) en pedazos consecutivos `<video>_s01.mp4`, `<video>_s02.mp4`, etc., descartando un último pedazo de menos de 2 segundos, y todos llevan como `video_origen` el nombre del video largo. Si la carpeta de videos no existe pero ya hay clips en `<salida>/clips/`, se usan esos con `video_origen` = `clip_id`. Después, para cada clip, extrae un fotograma cada `1/fps` segundos con OpenCV (se puede partir de la lógica por clip de `scripts/extract_frames.py`) a `<salida>/fotogramas/<clip_id>/frame_<NNN>.jpg`, llama a `analizar_frame(frame, modo=modo)` (importado dentro de `main`) y escribe una fila `fotograma` (`personas_reales`, `celulares_reales` y etiquetas vacías) seguida de una fila `persona` por cada elemento de `dato['personas']` (`origen: yolo`) y de `dato['poses_sin_asociar']` (`origen: solo_pose`, `detection_id` vacío). Si `etiquetas.csv` ya existe y no se pasa `--sobrescribir`, termina con un mensaje de error sin tocarlo (protege el etiquetado manual). Al final imprime cuántos videos originales, clips, fotogramas y filas de persona generó, y avisa si hay menos de 8 videos originales.
- **Test**: el plan no prevé archivo de pruebas para este script; se comprueba con la ejecución real. Comprobar a mano en una consola de Python que `iou([0,0,10,10],[0,0,10,10]) == 1.0`, `iou([0,0,10,10],[20,20,30,30]) == 0.0` y que `leer_etiquetas` devuelve `None` en celdas vacías.
- **Verify**: `python -m scripts.imagen_preparar_etiquetado --help` responde sin cargar YOLO; con videos en `datos_propios/videos`, `python -m scripts.imagen_preparar_etiquetado datos_propios/videos --fps 2` crea los clips de hasta 10 s en `datos_propios/clips/`, `datos_propios/fotogramas/<clip_id>/` y `datos_propios/etiquetas.csv`; una segunda ejecución sin `--sobrescribir` se niega; `git status` (o revisión del `.gitignore`) confirma que los fotogramas no se versionan y `etiquetas.csv` sí; `python -m pytest tests -q` sigue pasando.

### Paso 3: Calidad del fotograma, restauración y experimento (A, etapas 2 y 3)

- **Files**: crear `tests/test_nitidez.py`, `tests/test_restauracion.py`, `app/modulo_imagen/nitidez.py`, `app/modulo_imagen/restauracion.py`, `scripts/imagen_comparar_restauracion.py`.
- **Do**:
  - Pruebas primero (lista **Test**), con imágenes sintéticas NumPy/OpenCV al estilo de `tests/test_adaptativo.py`.
  - `nitidez.py`: `energia_alta_frecuencia(frame, radio_rel=0.10) -> float` (gris, `np.fft.fft2` + `fftshift`, energía `|F|²` fuera de un disco centrado de radio `radio_rel × min(alto, ancho)` dividida por la energía total sin la componente continua; 0.0 si la energía es 0); `desenfoque_movimiento(frame) -> {'anisotropia', 'direccion_grados'}` (Sobel X e Y en gris, `Ex = Σgx²`, `Ey = Σgy²`, `anisotropia = |Ex − Ey| / (Ex + Ey)`, dirección dominante con el tensor de estructura `0.5·atan2(2Σgxgy, Σgx² − Σgy²)` en grados; 0.0 y 0.0 si `Ex + Ey == 0`); `bloque_calidad(frame, *, umbral_nitidez=None) -> dict` que reutiliza `medir_calidad` sin editarla y devuelve `luminosidad`, `contraste`, `nitidez_laplaciana`, `energia_alta_frecuencia`, `ruido_mad_estimado`, `desenfoque_movimiento` y `apto_para_detalle` (`None` si `umbral_nitidez is None`, si no `nitidez_laplaciana >= umbral_nitidez`). Valores redondeados como en `calidad.py`.
  - `restauracion.py`: `restaurar(frame, calidad, *, modo='adaptativo') -> (imagen, bloque)`. Nunca muta la entrada (trabaja sobre copia). Con `modo='ninguno'` devuelve una copia idéntica y `operaciones == []`. Con `adaptativo`: `cv2.fastNlMeansDenoisingColored` si `calidad['ruido_mad_estimado']` supera `UMBRAL_RUIDO`; máscara de enfoque `original + cantidad × (original − GaussianBlur)` si `nitidez_laplaciana` está por debajo de `UMBRAL_NITIDEZ` y `desenfoque_movimiento.anisotropia` no supera `UMBRAL_ANISOTROPIA`. Los tres umbrales son constantes de módulo declaradas como provisionales en un comentario. `bloque = {'operaciones': [...], 'nitidez_antes': ..., 'nitidez_despues': ...}` con operaciones en el formato del contrato (`'nlm_h=6'`, `'mascara_enfoque=0.8'`); `modo` desconocido lanza `ValueError`.
  - `scripts/imagen_comparar_restauracion.py` (`--etiquetas`, `--fotogramas` por defecto `datos_propios/fotogramas`, `--salida` por defecto `runs/imagen/restauracion`, `--con-pose`, `--histograma-nitidez`): usa `leer_etiquetas`, `iou` y `emparejar_por_iou` del paso 2. Para cada fila `fotograma` con `personas_reales` rellenada, carga la imagen y genera tres variantes: original, `mejorar_imagen(original, modo='adaptativo')[0]` y `restaurar(mejorada, bloque_calidad(original))[0]`; en cada una llama a `detectar_imagen` (importado dentro de la función; devuelve `(anotada, detecciones)`) y, con `--con-pose`, a `analizar_postura`. Por variante cuenta personas encontradas `Σ min(detectadas, personas_reales) / Σ personas_reales`, fotogramas con celular encontrado sobre fotogramas con `celulares_reales > 0`, y, emparejando por IoU ≥ 0,5 contra las filas `persona`, cuántas personas etiquetadas se recuperan desglosado por `postura`; mide el tiempo por variante. Escribe `comparacion.md` con la tabla de tres columnas (conteos "x de y" junto al porcentaje) y la imprime. Con `--histograma-nitidez` guarda `histograma_nitidez.png` (matplotlib con backend `Agg`) de `nitidez_laplaciana` del original separado por `calidad_visual` (`nitida` / `borrosa`), para elegir a ojo el umbral de `apto_para_detalle`.
- **Test**: `test_nitidez.py`: tablero de ajedrez nítido frente al mismo con `GaussianBlur` → `energia_alta_frecuencia` del nítido mayor; tablero con borrón horizontal (núcleo 1 × 15 con `cv2.filter2D`) → `anisotropia` mayor que en el original; imagen uniforme → 0.0 en `energia_alta_frecuencia` y en `anisotropia`, sin error; `bloque_calidad` sin umbral → `apto_para_detalle is None` y con umbral muy bajo → `True`. `test_restauracion.py`: imagen limpia con ruido gaussiano añadido y `calidad` con `ruido_mad_estimado` muy por encima de `UMBRAL_RUIDO` → diferencia media absoluta respecto a la limpia menor que la de la ruidosa, la entrada no cambia (comparar con una copia previa) y `operaciones` contiene una entrada que empieza por `nlm`; `modo='ninguno'` → salida idéntica y `operaciones == []`; `calidad` con anisotropía alta y nitidez baja → no hay operación de máscara de enfoque.
- **Verify**: `python -m pytest tests/test_nitidez.py tests/test_restauracion.py -q` pasa (unas 7 pruebas); `python -m pytest tests -q` pasa; `python -m scripts.imagen_comparar_restauracion --help` responde sin cargar YOLO; con datos etiquetados, `python -m scripts.imagen_comparar_restauracion --etiquetas datos_propios/etiquetas.csv --histograma-nitidez` escribe `runs/imagen/restauracion/comparacion.md` y el histograma.

### Paso 4: Escena calibrada y relaciones persona–objeto–pasamanos (B, etapas 4 y 5)

- **Files**: crear `tests/test_escena.py`, `tests/test_relaciones.py`, `app/modulo_imagen/escena.py`, `app/modulo_imagen/relaciones.py`, `scripts/imagen_calibrar_escena.py`.
- **Do**:
  - Pruebas primero (lista **Test**). Ningún archivo importa `analisis_tiempo_real`, `detector` ni `pose`.
  - Formato de `configs/escena/<camara_id>.json`: `{'camara_id': str, 'ancho_ref': int, 'alto_ref': int, 'zona_escalones': [[x, y], ...], 'pasamanos': [x1, y1, x2, y2] | None}`.
  - `escena.py`: `cargar_calibracion(camara_id, carpeta=Path('configs/escena'))` (dict o `None` si no existe el archivo); `escalar_calibracion(calibracion, ancho, alto)` (devuelve una copia con polígono y segmento multiplicados por `ancho/ancho_ref` y `alto/alto_ref`, redondeados a int, y `ancho_ref`/`alto_ref` actualizados); `contar_lineas_escalon(frame, zona)` (gris, Canny, cierre morfológico, máscara del polígono, `cv2.HoughLinesP`, filtrar líneas largas, quedarse con las de orientación parecida a la dominante, y **fusionar** segmentos de orientación parecida cuya distancia perpendicular sea menor que `max(6, 0.02 × alto de la zona)` antes de contar); `punto_en_zona(punto, zona)` con `cv2.pointPolygonTest` (`>= 0` cuenta como dentro); `bloque_escena(frame, calibracion)`: si `calibracion is None` devuelve `{'origen_zona': 'no_disponible', 'zona_escalones': None, 'lineas_escalon_detectadas': None, 'pasamanos': None}`; si no, escala la calibración al tamaño del fotograma y devuelve `origen_zona: 'calibracion_manual'`, la zona escalada, el conteo de líneas y `pasamanos: {'visible': segmento is not None, 'segmento': segmento}` (en esta versión `visible` solo indica que el pasamanos está calibrado).
  - `relaciones.py`: lectura propia de muñecas (índices COCO 9 = izquierda y 10 = derecha de `persona['pose']['puntos_corporales']`, confianza ≥ 0,5, igual que `_conf_punto` pero sin importarlo); altura de la persona = `bbox[3] − bbox[1]` (mínimo 1). `objetos_en_mano(persona, objetos, *, umbral_rel=0.18) -> list[dict]`: para cada objeto cuya `clase` no sea `person`, distancia del centro de su caja a cada muñeca visible dividida por la altura; si la menor es ≤ `umbral_rel` añade `{'indice_objeto': i, 'clase', 'mano': 'izquierda' | 'derecha', 'confianza_detector', 'distancia_relativa'}` (acepta objetos con `confianza` o `confianza_detector`); sin muñecas visibles devuelve `[]`. `contacto_pasamanos(persona, segmento, *, umbral_rel=0.12) -> str`: `'no_evaluable'` si `segmento is None` o no hay muñecas visibles; si no, distancia punto–segmento mínima entre muñecas dividida por la altura, `'si'` si es menor que `umbral_rel` y `'no'` en caso contrario. Comentario: los `umbral_rel` por defecto son provisionales y se sustituirán por los elegidos con `toca_pasamanos` cuando haya etiquetas.
  - `scripts/imagen_calibrar_escena.py`: `python -m scripts.imagen_calibrar_escena foto.jpg --camara <camara_id> [--carpeta configs/escena]`. Ventana OpenCV con `setMouseCallback`: clics izquierdos añaden esquinas del polígono; la tecla `p` pasa a marcar los dos extremos del pasamanos; `z` deshace el último punto; `Enter` guarda; `Esc` sale sin guardar. Dibuja lo marcado en vivo y guarda el JSON con el tamaño de la foto como referencia. Rechaza guardar con menos de 3 esquinas o con un pasamanos de un solo punto.
- **Test**: `test_escena.py`: imagen negra con 6 líneas blancas horizontales dentro del polígono → `contar_lineas_escalon` entre 5 y 7; una sola franja blanca de 4 px de grosor → exactamente 1; punto dentro y fuera del polígono → `True` y `False`; calibración de 1280 × 720 escalada a 640 × 360 → coordenadas a la mitad; `bloque_escena(frame, None)['origen_zona'] == 'no_disponible'`; `cargar_calibracion` con `tmp_path` vacío → `None` y con un JSON escrito → el mismo dict. `test_relaciones.py` (persona de 300 px de alto con pose sintética): muñeca a 5 px del segmento → `'si'`; muñeca a 150 px → `'no'`; sin muñecas visibles (confianza 0,1) o con `segmento=None` → `'no_evaluable'`; celular centrado en la muñeca derecha (punto 10) → un solo elemento con `mano == 'derecha'`; objeto lejano → `[]`.
- **Verify**: `python -m pytest tests/test_escena.py tests/test_relaciones.py -q` pasa (unas 11 pruebas); `python -m pytest tests -q` pasa; `python -m scripts.imagen_calibrar_escena --help` responde; con la foto de la escalera vacía, `python -m scripts.imagen_calibrar_escena datos_propios/escalera_vacia.jpg --camara escalera_g` crea `configs/escena/escalera_g.json`.

### Paso 5: Clasificador de postura propio HOG + LBP + SVM (C, etapa 6)

- **Files**: crear `tests/test_descriptores.py`, `tests/test_clasificador_postura.py`, `app/modulo_imagen/descriptores.py`, `app/modulo_imagen/clasificador_postura.py`, `scripts/imagen_entrenar_postura.py`.
- **Do**:
  - Pruebas primero (lista **Test**).
  - `descriptores.py`: `recorte_persona(frame, bbox, tam=(64, 128))`: recorta la caja recortada a los límites de la imagen, pasa a gris y redimensiona a 64 de ancho × 128 de alto (`uint8`, forma `(128, 64)`); `None` si la caja es degenerada (ancho o alto ≤ 1 tras recortar) o cae fuera de la imagen. `vector_hog_lbp(recorte) -> np.ndarray float32` de longitud 3790: `skimage.feature.hog` con `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` **pasado explícitamente** (el valor por defecto da 6804) → 3780 valores; más histograma normalizado de `local_binary_pattern(P=8, R=1, method='uniform')` con 10 bins → 10 valores.
  - `clasificador_postura.py`: `CLASES = ('erguida', 'inclinada', 'agachada_o_sentada', 'horizontal')`; `crear_pipeline(C=1.0, gamma='scale')` → `Pipeline([StandardScaler(), PCA(n_components=0.95), SVC(kernel='rbf', class_weight='balanced', C=C, gamma=gamma)])`; `entrenar(X, y, grupos) -> (pipeline, registro)`: `GridSearchCV` pequeño sobre `svc__C` y `svc__gamma` con `GroupKFold(n_splits=min(5, n_grupos))`; si hay menos de 2 grupos, ajusta `crear_pipeline()` sin búsqueda y lo anota en el registro; `registro` con mejores parámetros, puntuación de validación cruzada (F1 macro), número de filas, de grupos y conteo por clase. `cargar_modelo(ruta=Path('runs/imagen/postura_svm.joblib'))` → `joblib.load` o `None` si no existe. `predecir_postura(frame, bbox, modelo)`: `'no_disponible'` si `modelo is None`; `'no_evaluable'` si `recorte_persona` devuelve `None`; si no, la clase predicha como `str`.
  - `scripts/imagen_entrenar_postura.py` (`--etiquetas`, `--fotogramas` por defecto `datos_propios/fotogramas`, `--clips-prueba` por defecto `datos_propios/clips_prueba.txt`, `--escalera-prueba` opcional, `--variante original|restaurada` por defecto `original`, `--salida` por defecto `runs/imagen/postura_svm.joblib`, `--semilla` por defecto 42): usa `leer_etiquetas` del paso 2. Toma las filas `persona` con `postura` en `CLASES`; las `no_se_ve` se excluyen y se cuentan. División por `video_origen` (nunca por pedazo ni por fotograma): si existe `clips_prueba.txt` se usa tal cual; si no, se eligen los videos originales de la escalera indicada en `--escalera-prueba` (prefijo del `video_origen`) o, sin ella, el 20 % de los videos originales al azar con la semilla (al menos uno), y se escribe el archivo con un `video_origen` por línea. Todos los clips de esos videos son de prueba y no se tocan. El grupo que recibe `entrenar` es el `video_origen`. Recorte con la `bbox` de la hoja sobre el fotograma (con `--variante restaurada`, sobre `restaurar(mejorar_imagen(img)[0], bloque_calidad(img))[0]`, importados dentro de la función); aumento solo en entrenamiento: volteo horizontal y brillo ±15 %, cada copia con el grupo de su video original. Llama a `entrenar`, guarda el modelo con `joblib.dump` y un registro JSON junto a él (`runs/imagen/postura_svm.json`) con `variante`, filas y clips usados, filas `no_se_ve` excluidas, clips de prueba y el `registro` de `entrenar`; lo imprime. La variante guardada es la que el paso 6 usará en uso real, para que entrenamiento y uso coincidan.
- **Test**: `test_descriptores.py`: recorte válido → forma `(128, 64)` y `dtype uint8`; `vector_hog_lbp` → longitud 3790, `float32` y todo finito; caja de ancho cero → `None`; caja totalmente fuera de la imagen → `None`. `test_clasificador_postura.py`: datos sintéticos de 40 recortes (rectángulo blanco vertical frente a horizontal sobre fondo negro, con leve ruido) en 8 grupos, pasados por `vector_hog_lbp` → `entrenar` devuelve un pipeline que acierta más de 90 % en entrenamiento; `cargar_modelo(tmp_path / 'no_existe.joblib') is None` y `predecir_postura(frame, bbox, None) == 'no_disponible'`; con un modelo entrenado y caja degenerada → `'no_evaluable'`.
- **Verify**: `python -m pytest tests/test_descriptores.py tests/test_clasificador_postura.py -q` pasa (unas 7 pruebas, en pocos segundos: mantener la rejilla pequeña); `python -m pytest tests -q` pasa; `python -m scripts.imagen_entrenar_postura --help` responde; con datos etiquetados, `python -m scripts.imagen_entrenar_postura --etiquetas datos_propios/etiquetas.csv` crea `runs/imagen/postura_svm.joblib`, `runs/imagen/postura_svm.json` y, la primera vez, `datos_propios/clips_prueba.txt`.

### Paso 6: Integración de las piezas en `analizar_clip` y `evidencia_de_clip` (A, etapa 8, primera parte)

Requiere los pasos 1, 3, 4 y 5.

- **Files**: crear `tests/test_clip_enriquecer.py`; modificar `tests/test_evidencia.py`, `app/modulo_imagen/analizador_clip.py`, `app/modulo_imagen/evidencia.py`.
- **Do**:
  - `tests/test_clip_enriquecer.py`: mismo patrón que `tests/test_clip_seguimiento_etapa4.py` (módulo `analisis_tiempo_real` falso en `sys.modules`, `analizar_frame`, `recuperar_en_roi` y `recuperar_pose_en_roi` simulados con `monkeypatch`, clip sintético `.avi` en `tmp_path`).
  - `analizador_clip.py`: `analizar_clip` gana el parámetro de palabra clave `enriquecer=None`; si se pasa, se llama `enriquecer(original, dato)` justo antes de `agregar_senales(dato)`, solo para fotogramas decodificados. Nada más cambia.
  - `evidencia.py`: `crear_enriquecedor(*, modelo_postura=None, calibracion=None, modo='adaptativo', variante_postura='original', umbral_nitidez=None, deteccion_doble=False, detector=None)` devuelve una función `(original, dato) -> None` que escribe `dato['evidencia_imagen']` con la forma acordada en el paso 1: `calidad = bloque_calidad(original, umbral_nitidez=...)`; `restauracion` = bloque de `restaurar(mejorar_imagen(original, modo=modo)[0], calidad)`; `calibracion` = `escalar_calibracion` al tamaño del fotograma (o `None`); `escena = bloque_escena(original, calibracion)`; `posturas_clasificador` = `predecir_postura(variante, p['bbox'], modelo_postura)` para cada `p` de `personas_de_dato(dato)`, donde `variante` es el original o el restaurado según `variante_postura`. No toca ninguna clave existente de `dato`. Las importaciones de `preprocesamiento`, `nitidez`, `restauracion`, `escena`, `relaciones` y `clasificador_postura` pueden ir en cabecera (no cargan YOLO). **Detección doble:** si `deteccion_doble` es `True`, el enriquecedor llama a `detector(restaurada)` (por defecto `detectar_imagen`, importado dentro de la función para no cargar YOLO al importar el módulo; devuelve `(anotada, detecciones)`) y escribe `deteccion_restaurada = {'personas': n, 'objetos': m, 'solo_tras_restaurar': [...]}`, donde `solo_tras_restaurar` son las detecciones de la segunda pasada (con `clase`, `bbox`, `confianza_detector`) sin pareja de la misma clase con IoU ≥ 0,5 entre las cajas de `dato['personas']`, `dato['celulares']` y `dato['otros_objetos']`. Si `restaurar` no aplicó ninguna operación, no se repite la detección: se escriben los conteos de la pasada normal y `solo_tras_restaurar: []`. Con `deteccion_doble=False`, `deteccion_restaurada` es `None`. No se edita `analizar_frame`.
  - `construir_fotograma` completa los campos que el paso 1 dejó provisionales: `en_zona_escalones` de personas y objetos con `punto_en_zona` del punto medio de la base de la caja si hay zona calibrada (si no, `None`); `contacto_pasamanos` con el segmento de `evidencia_imagen['calibracion']`; `objetos_en_mano` con `objetos_en_mano(persona, objetos)` para cada persona y, si un mismo `indice_objeto` aparece en dos personas, se conserva solo en la de menor `distancia_relativa`, se pone el `detection_id` de esa persona en `objetos[i].asociado_a_persona` y se quita `indice_objeto` de la salida.
  - `evidencia_de_clip(ruta_clip, *, clip_id=None, camara_id='camara_local', contexto_video=None, modo='adaptativo', intervalo_s=1.0, modelo_postura=None, calibracion=None, deteccion_doble=None) -> dict`: `deteccion_doble=None` significa usar la constante `DETECCION_DOBLE`; si `calibracion is None`, añade a `limitaciones` "Sin calibración para la cámara '<camara_id>'"; importa `analizar_clip` dentro de la función, lo llama con `enriquecer=crear_enriquecedor(...)` y `contexto_video`, construye la evidencia con `clip_id or Path(ruta_clip).stem`, añade a `limitaciones` la frase del clasificador (con filas y clips leídos de `runs/imagen/postura_svm.json` si existe; si `modelo_postura is None`, que el clasificador no estaba disponible), llama a `validar_evidencia` y devuelve el dict. No carga por su cuenta ni modelo ni calibración: los pasa quien llama.
- **Test**: `test_clip_enriquecer.py`: con un `enriquecer` que registra llamadas y añade `dato['evidencia_imagen']`, se llama una vez por fotograma decodificado y recibe el fotograma original (comparar forma y contenido con el del clip sintético); sin `enriquecer`, el resultado tiene las mismas claves y el mismo `resumen_visual` que antes. Añadir a `test_evidencia.py`: un celular cerca de las muñecas de dos personas → solo figura en `objetos_en_mano` de la más cercana y `asociado_a_persona` es su `detection_id`; con `evidencia_imagen.calibracion` que tiene zona y pasamanos, una persona con los pies dentro del polígono tiene `en_zona_escalones is True` y una muñeca junto al pasamanos da `contacto_pasamanos == 'si'`; sin calibración, `en_zona_escalones is None` y `contacto_pasamanos == 'no_evaluable'`; `crear_enriquecedor(deteccion_doble=True, detector=falso)` con un detector falso que devuelve una persona ya presente en `dato` y un `cell phone` nuevo, sobre una imagen ruidosa que sí se restaura, escribe `deteccion_restaurada` con `solo_tras_restaurar` de un solo elemento de clase `cell phone`, y con `deteccion_doble=False` la deja en `None` sin llamar al detector; `evidencia_de_clip` sin calibración incluye en `limitaciones` la frase de la cámara; `crear_enriquecedor()` aplicado a una imagen sintética y un `dato` mínimo escribe `evidencia_imagen` con `posturas_clasificador == ['no_disponible']` y sin modificar las demás claves; `evidencia_de_clip` con `app.modulo_imagen.analizador_clip` falso en `sys.modules` (devuelve un resultado de clip mínimo) produce un JSON que valida y cuyo `clip.clip_id` es el nombre del archivo sin extensión.
- **Verify**: `python -m pytest tests/test_clip_enriquecer.py tests/test_evidencia.py -q` pasa; `python -m pytest tests -q` pasa completo, en particular `test_clip_seguimiento_etapa4.py` y `test_clip_rescate_pose_nuevo.py` sin modificarse.

### Paso 7: Ruta HTTP de evidencia, demo y documentación (A, etapa 8, segunda parte)

Requiere el paso 6.

- **Files**: crear `tests/test_http_evidencia.py`, `scripts/imagen_demo.py`; modificar `app/integracion_imagen.py`, `MODULO_IMAGEN_PASO1.md`.
- **Do**:
  - `tests/test_http_evidencia.py`: mismo montaje que `tests/test_http_integracion_video_imagen.py` (fixture `web` con `pytest.importorskip('flask')`, módulo falso `app.modulo_imagen.evidencia` en `sys.modules` cuyo `evidencia_de_clip` acepta `**kwargs`, registra la ruta del temporal y devuelve el fixture con el `clip_id` recibido).
  - `integracion_imagen.py`: extraer las validaciones de `recibir_clip` (tamaño, `_archivo('clip', ...)`, `_configuracion('clip')`, `intervalo_s` entre 0,1 y 5,0, extensión) a `_leer_peticion_clip() -> (contenido, metadatos, modo, intervalo, extension)` que lanza las mismas excepciones; `recibir_clip` lo usa sin cambiar sus respuestas (`incluir_muestras` sigue leyéndose en `recibir_clip`). Ruta nueva `POST /api/imagen/v1/evidencia` en el mismo blueprint: usa `_leer_peticion_clip`, devuelve `_error_conocido` en errores de validación, escribe el temporal con `tempfile.NamedTemporaryFile(suffix=extension, delete=False)`, importa dentro de la función `evidencia_de_clip`, `cargar_modelo` y `cargar_calibracion`, llama a `evidencia_de_clip(ruta, clip_id=metadatos['clip_id'], camara_id=metadatos['camara_id'], contexto_video=metadatos['contexto_video'], modo=modo, intervalo_s=intervalo, modelo_postura=cargar_modelo(), calibracion=cargar_calibracion(metadatos['camara_id']))`, responde el JSON tal cual con 200 y borra el temporal en `finally`. Las rutas `/salud`, `/fotograma` y `/clip` no cambian.
  - `scripts/imagen_demo.py`: `python -m scripts.imagen_demo clip.mp4 [--camara escalera_g] [--modelo runs/imagen/postura_svm.joblib] [--intervalo 1.0] [--salida runs/imagen/demo]`. Carga modelo y calibración, llama a `analizar_clip` con `enriquecer=crear_enriquecedor(...)` y `guardar_en` una carpeta temporal (para tener `frame_XXX_original.jpg` y `frame_XXX_anotado.jpg`), construye y valida la evidencia con `construir_evidencia` y la guarda como `<salida>/<clip_id>/evidencia.json`; genera `lamina.jpg` con una fila por fotograma y tres vistas: original, restaurado (recalculado con `restaurar(mejorar_imagen(original)[0], bloque_calidad(original))`) y anotado con polígono de escalones y pasamanos dibujados encima de la imagen anotada existente (cajas y esqueleto). Imprime la ruta de ambos archivos.
  - `MODULO_IMAGEN_PASO1.md`: un párrafo que remite a `plans/2026-11-14_modulo-imagen-evidencia.md`, al contrato `imagen-evidencia-1.0` y a la ruta nueva.
- **Test** (`tests/test_http_evidencia.py`): POST con un `.avi` pequeño y `metadatos` con `clip_id` → 200, `schema_version == 'imagen-evidencia-1.0'`, `clip.clip_id` igual al de los metadatos y el temporal ya no existe; extensión `.txt` → 400; `intervalo_s=9` → 400.
- **Verify**: `python -m pytest tests/test_http_evidencia.py tests/test_http_integracion_video_imagen.py -q` pasa (la segunda sin modificarse, demuestra que `/clip` responde igual); `python -m pytest tests -q` pasa completo; `python -m scripts.imagen_demo --help` responde; con un clip real, `python -m scripts.imagen_demo datos_propios/clips/g2_1117_a1_celular_02.mp4 --camara escalera_g` produce un `evidencia.json` que `validar_evidencia` acepta y una lámina de tres vistas.

### Paso 8: Evaluación sobre clips de prueba e informe (C, etapa 7)

Requiere los pasos 2, 5 y 6 (no el 7).

- **Files**: crear `scripts/imagen_evaluar.py`; tras la ejecución con datos reales, crear `informes/imagen_evaluacion.md` (y sus figuras) como copia versionada del informe.
- **Do**:
  - `scripts/imagen_evaluar.py` (`--etiquetas`, `--clips` por defecto `datos_propios/clips`, `--clips-prueba` por defecto `datos_propios/clips_prueba.txt`, `--camara` opcional para cargar la calibración, `--modelo` por defecto `runs/imagen/postura_svm.joblib`, `--intervalo` por defecto 0,5 para coincidir con la extracción a 2 fps, `--salida` por defecto `runs/imagen/evaluacion`, `--copiar-informe`): usa `leer_etiquetas` y `emparejar_por_iou` del paso 2. Solo procesa los clips cuyo `video_origen` (según la hoja) está en `clips_prueba.txt` (si falta, error: la división la fija el paso 5). Para cada clip llama a `evidencia_de_clip` con modelo y calibración, midiendo el tiempo con `time.perf_counter`; empareja cada fotograma con las filas de la hoja por `clip_id` y `segundo_s` (tolerancia 0,05 s) y cada persona por IoU ≥ 0,5.
  - Métricas, siempre con conteos "x de y" junto al porcentaje: postura de la regla (mapeada con la correspondencia de la sección 4; `agachada_o_sentada` nunca acertada, anotado como limitación de la línea base) y del SVM: matriz de confusión, recall por clase y F1 macro (`sklearn.metrics`), excluyendo y contando `no_se_ve`, y la exactitud de un modelo que siempre dice `erguida` como referencia; celular en mano (fila con `objeto_en_mano == 'celular'` frente a `cell phone` en `objetos_en_mano`): precisión y recall; contacto con pasamanos: recall de `no` y exactitud sobre filas con `toca_pasamanos` en `si`/`no`, contando aparte `no_evaluable`; calidad: mediana y rango de `nitidez_laplaciana` para `nitida` y `borrosa` con histograma; tiempo: ms por fotograma y por clip, indicando si la detección doble estaba encendida; si lo estaba, cuántas detecciones aparecieron `solo_tras_restaurar` y en cuántos fotogramas. Personas etiquetadas sin emparejar se cuentan como no detectadas.
  - Salida: `runs/imagen/evaluacion/informe.md` con tablas, figuras PNG (matplotlib `Agg`: matrices de confusión de regla y SVM, histograma de nitidez) y una lista de errores con la ruta del fotograma (sin incrustar imágenes de personas) y la clase real y predicha. Con `--copiar-informe`, copia el informe a `informes/imagen_evaluacion.md` y las figuras a `informes/` con prefijo `imagen_evaluacion_`, reescribiendo los enlaces.
- **Test**: el plan no prevé archivo de pruebas para este script; se comprueba con la ejecución real sobre los clips de prueba, revisando a mano que la suma de cada matriz de confusión coincide con el número de filas evaluadas y que ningún clip de entrenamiento aparece en el informe.
- **Verify**: `python -m scripts.imagen_evaluar --help` responde sin cargar YOLO; `python -m pytest tests -q` sigue pasando; con datos y modelo, `python -m scripts.imagen_evaluar --etiquetas datos_propios/etiquetas.csv --camara escalera_g --copiar-informe` crea `runs/imagen/evaluacion/informe.md` con matriz de confusión sobre clips de prueba y su copia en `informes/imagen_evaluacion.md`. Con esto se cumple el "Hecho cuando" de la sección 11, salvo la confirmación de Predicción y las preguntas de defensa.
