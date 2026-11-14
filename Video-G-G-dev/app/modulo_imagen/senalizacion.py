"""Señales visuales explicables del módulo Imagen.

Solo utiliza detecciones efectivamente reportadas por los modelos. Las señales
son pistas para revisar, no clasificaciones verificadas de caídas ni de marcha.
"""

POSTURAS_REVISION = {'postura_horizontal_posible', 'inclinacion_para_revision'}


def _senal(codigo, nivel, titulo, detalle):
    return {'codigo': codigo, 'nivel': nivel, 'titulo': titulo, 'detalle': detalle}


def senales_fotograma(dato):
    """Calcula observaciones usando SOLO evidencia existente en el JSON."""
    senales = []
    personas = dato.get('personas') or []
    poses = dato.get('poses_sin_asociar') or []
    roir = dato.get('recuperacion_roi') or {}
    posturas = [p.get('estado_postural') for p in personas + poses]
    if roir.get('origen') == 'roi_pose':
        posturas.append(roir.get('estado_postural'))
    if 'postura_horizontal_posible' in posturas:
        senales.append(_senal('postura_horizontal', 'revision', 'Posible incidente: ATENCIÓN PRIORITARIA, postura horizontal',
                              'Se observó una postura horizontal o un torso muy inclinado. Una imagen aislada no confirma caída.'))
    elif 'inclinacion_para_revision' in posturas:
        senales.append(_senal('inclinacion', 'revision', 'Inclinación corporal: revisar',
                              'El ángulo o la geometría de la pose ameritan revisión. Caminar en escaleras también puede producir inclinación normal.'))
    continuidad = dato.get('continuidad_visual') or {}
    if continuidad.get('estado') == 'seguimiento_indicio_previo':
        senales.append(_senal('seguimiento_indicio_previo', 'revision',
                              'ALERTA EN SEGUIMIENTO: postura sospechosa anterior',
                              f"Última evidencia hace {continuidad.get('segundos_desde_evidencia', 0)} s. "
                              'No hay confirmación de que la persona siga en la misma postura.'))
    if dato.get('revision_incidente') == 'deteccion_perdida_tras_indicio':
        senales.append(_senal('perdida_tras_indicio', 'revision', 'Se perdió una observación sospechosa',
                              'La pose anterior requería revisión y la persona no fue localizada en este fotograma.'))
    elif dato.get('revision_incidente') == 'perdida_visual_no_concluyente':
        senales.append(_senal('perdida_visual', 'observacion', 'Persona no localizada en el siguiente fotograma',
                              'Podría haber salido de escena, quedado tapada o fallado el detector. No demuestra caída.'))
    if roir.get('origen'):
        senales.append(_senal('rescate_roi', 'observacion', 'Búsqueda ROI con evidencia visual',
                              f"Se obtuvo una observación nueva con {roir['origen']}. La identidad requiere confirmación temporal."))
    telefonos = dato.get('celulares') or []
    if telefonos:
        cercano = any((t.get('relacion_manos') or {}).get('estado') == 'cerca_de_mano' for t in telefonos)
        senales.append(_senal('celular', 'revision' if cercano else 'observacion',
                              'Posible celular cerca de la mano: revisar' if cercano else 'Celular detectado',
                              f'{len(telefonos)} detección(es) del modelo; '
                              'detectar el objeto no demuestra distracción. La cercanía geométrica no confirma uso. '
                              'Las detecciones de recortes ROI son exploratorias.'))
    pre = dato.get('preprocesamiento') or {}
    antes = pre.get('antes') or dato.get('calidad_original') or {}
    if antes.get('brillo_bajo_estimado') or antes.get('contraste_bajo_estimado') or antes.get('sombras_fuertes_estimadas'):
        problemas = []
        if antes.get('brillo_bajo_estimado'): problemas.append('poca luz')
        if antes.get('contraste_bajo_estimado'): problemas.append('contraste bajo')
        if antes.get('sombras_fuertes_estimadas'): problemas.append('sombras estimadas')
        senales.append(_senal('calidad', 'observacion', 'Limitación de calidad visual',
                              'Se observaron: ' + ', '.join(problemas) + '. Puede disminuir la detección.'))
    if not senales:
        senales.append(_senal('sin_indicio', 'informativo', 'Sin indicios detectados',
                              'No se reportó una señal con las reglas actuales; esto no demuestra que la escena sea segura.'))
    return senales


def agregar_senales(dato):
    dato['senales_visuales'] = senales_fotograma(dato)
    dato['nivel_revision_visual'] = ('revision' if any(s['nivel'] == 'revision' for s in dato['senales_visuales'])
                                     else 'observacion' if any(s['nivel'] == 'observacion' for s in dato['senales_visuales'])
                                     else 'informativo')
    dato['marcha'] = {'estado': 'no_evaluada',
                     'motivo': 'Imagen no estima marcha inestable sin seguimiento temporal fiable e identificación de personas.'}
    return dato


def resumen_clip(resultados):
    validos = [r for r in resultados if r.get('estado') != 'no_decodificado']
    def tiene(r, codigo):
        return any(s.get('codigo') == codigo for s in r.get('senales_visuales', []))
    return {
        'fotogramas_validos': len(validos),
        'fotogramas_revision': sum(r.get('nivel_revision_visual') == 'revision' for r in validos),
        'fotogramas_observacion': sum(r.get('nivel_revision_visual') == 'observacion' for r in validos),
        'detecciones_personas_por_fotograma': sum(len(r.get('personas', [])) for r in validos),
        'detecciones_celulares_por_fotograma': sum(len(r.get('celulares', [])) for r in validos),
        'detecciones_otros_objetos_por_fotograma': sum(len(r.get('otros_objetos', [])) for r in validos),
        'fotogramas_calidad_limitada': sum(tiene(r, 'calidad') for r in validos),
        'rescates_roi': sum(bool((r.get('recuperacion_roi') or {}).get('origen')) for r in validos),
        'fotogramas_alerta_seguimiento': sum((r.get('continuidad_visual') or {}).get('estado') == 'seguimiento_indicio_previo' for r in validos),
        'marcha_evaluada': False,
        'nota': 'Los conteos son detecciones por fotograma, no personas ni objetos únicos; no confirma accidentes.'
    }
