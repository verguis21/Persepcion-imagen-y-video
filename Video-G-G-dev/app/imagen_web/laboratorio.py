"""Visor experimental con resultados REALES del módulo Imagen.

No consulta Video ni Audio; evita exponer como mediciones las cifras ficticias
presentes en html_export.zip. Se monta únicamente en el servidor de Imagen.
"""
from flask import Blueprint, render_template

laboratorio_bp = Blueprint('laboratorio_imagen', __name__, template_folder='../templates')

@laboratorio_bp.get('/imagen/laboratorio')
def laboratorio_imagen():
    return render_template('laboratorio_imagen.html')
