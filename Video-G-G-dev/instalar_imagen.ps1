# Instalador del entorno del modulo Imagen (Windows, PowerShell).
#
# Uso, desde la carpeta Video-G-G-dev:
#   .\instalar_imagen.ps1
#
# 1. Crea el entorno .venv_imagen con Python 3.11 si no existe.
# 2. Instala requirements-imagen.txt (NO el requirements.txt general de Video).
# 3. Corre toda la bateria de pruebas.
# Se puede volver a ejecutar sin romper nada: reutiliza el entorno existente.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$python = '.\.venv_imagen\Scripts\python.exe'
$paso = 'inicio'

# Si un comando no existe (por ejemplo, falta el lanzador `py`), PowerShell
# lanza una excepcion; se informa igualmente el paso que fallo.
trap {
    Write-Host "ERROR en el paso: $paso ($($_.Exception.Message))" -ForegroundColor Red
    exit 1
}

function Fallar([string]$paso) {
    Write-Host "ERROR en el paso: $paso (codigo $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $python)) {
    $paso = 'crear entorno (py -3.11 -m venv .venv_imagen)'
    Write-Host 'Creando entorno .venv_imagen con Python 3.11...'
    py -3.11 -m venv .venv_imagen
    if ($LASTEXITCODE -ne 0) { Fallar 'crear entorno (py -3.11 -m venv .venv_imagen)' }
} else {
    Write-Host 'El entorno .venv_imagen ya existe; se reutiliza.'
}

$paso = 'instalar dependencias (pip install -r requirements-imagen.txt)'
Write-Host 'Instalando requirements-imagen.txt...'
& $python -m pip install -r requirements-imagen.txt
if ($LASTEXITCODE -ne 0) { Fallar 'instalar dependencias (pip install -r requirements-imagen.txt)' }

$paso = 'pruebas (python -m pytest tests -q)'
Write-Host 'Ejecutando pruebas...'
& $python -m pytest tests -q
if ($LASTEXITCODE -ne 0) { Fallar 'pruebas (python -m pytest tests -q)' }

Write-Host 'Entorno de Imagen listo' -ForegroundColor Green
