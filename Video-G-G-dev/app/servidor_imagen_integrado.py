r"""Arranque experimental aislado para Imagen + puente v1 (Video permanece intacto).

Desde la raiz del proyecto, en Windows:
    .\.venv_imagen\Scripts\python.exe -m app.servidor_imagen_integrado
"""
from flask import Flask
from .imagen_pruebas import imagen_bp
from .integracion_imagen import integracion_bp
from .laboratorio_imagen import laboratorio_bp

app = Flask(__name__)
# El endpoint antiguo sigue limitando las fotografias a 7MB por su cuenta.
# El puente permite clips de hasta 50MB, siempre con validacion por ruta.
app.config['MAX_CONTENT_LENGTH'] = 51 * 1024 * 1024
app.register_blueprint(imagen_bp)
app.register_blueprint(integracion_bp)
app.register_blueprint(laboratorio_bp)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5051, debug=False, threaded=False)
