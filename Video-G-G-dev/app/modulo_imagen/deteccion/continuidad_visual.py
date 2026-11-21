"""Seguimiento de INDICIOS por ventana temporal; no es tracking de personas.

Una señal no desaparece inmediatamente por la pérdida de una detección, pero
nunca se afirma que la persona siga en el suelo sin evidencia actual.
"""
from __future__ import annotations

POSTURAS_RELEVANTES = ('postura_horizontal_posible', 'inclinacion_para_revision')


class ContinuidadVisual:
    def __init__(self, persistencia_s=2.0):
        self.persistencia_s = max(0.0, float(persistencia_s))
        self.ultimo_indicio = None

    def aplicar(self, resultado):
        instante = resultado.get('segundo_s')
        if instante is None:
            return resultado
        t = float(instante)
        candidatos = list(resultado.get('personas') or []) + list(resultado.get('poses_sin_asociar') or [])
        roi = resultado.get('recuperacion_roi') or {}
        if roi.get('origen'):
            candidatos.append(roi)
        visibles = [p for p in candidatos if p.get('estado_postural') in POSTURAS_RELEVANTES]
        if visibles:
            estado = ('postura_horizontal_posible' if any(p['estado_postural'] == 'postura_horizontal_posible' for p in visibles)
                      else 'inclinacion_para_revision')
            self.ultimo_indicio = {'segundo_s': t, 'tipo': estado}
            resultado['continuidad_visual'] = {
                'estado': 'indicio_actual', 'tipo': estado,
                'evidencia_actual': True, 'segundos_desde_evidencia': 0.0,
                'nota': 'Indicio calculado en el fotograma actual; no confirma un accidente.'
            }
        elif self.ultimo_indicio:
            lapso = t - self.ultimo_indicio['segundo_s']
            if 0 < lapso <= self.persistencia_s:
                resultado['continuidad_visual'] = {
                    'estado': 'seguimiento_indicio_previo', 'tipo': self.ultimo_indicio['tipo'],
                    'evidencia_actual': False, 'segundos_desde_evidencia': round(lapso, 3),
                    'nota': 'Alerta visual temporal por una postura anterior. No demuestra que la persona permanezca en el suelo.'
                }
                if resultado.get('revision_incidente') in (None, 'sin_indicio_suficiente', 'sin_evidencia_actual'):
                    resultado['revision_incidente'] = 'seguimiento_indicio_previo'
            elif lapso > self.persistencia_s or lapso < 0:
                self.ultimo_indicio = None
        return resultado
