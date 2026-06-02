@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0
set API_URL=http://localhost:8000
set BACKEND_PORT=8000

:: ── Argumentos ────────────────────────────────────────────────────
:parse
if "%~1"=="" goto :end_parse
if "%~1"=="--api-url" (
    set API_URL=%~2
    shift /1
    shift /1
    goto :parse
)
if "%~1"=="--port" (
    set BACKEND_PORT=%~2
    shift /1
    shift /1
    goto :parse
)
if "%~1"=="--help" goto :usage
if "%~1"=="-h" goto :usage
echo Argumento desconocido: %~1
goto :usage

:end_parse

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║     SteamAgent Recommender — Launcher     ║
echo   ╚══════════════════════════════════════════╝
echo.

:: ── Requisitos ────────────────────────────────────────────────────
echo [1/5] Verificando requisitos...

set PYTHON_CMD=python
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=py -3
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo Error: Python no encontrado. Instalalo desde https://www.python.org/
        echo   y deshabilita los alias de Microsoft Store en:
        echo   Configuracion ^> Aplicaciones ^> Alias de ejecucion de aplicaciones
        exit /b 1
    )
)

where flutter >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Flutter no encontrado. Instalalo desde https://flutter.dev/
    exit /b 1
)

:: ── Backend ───────────────────────────────────────────────────────
echo [2/5] Instalando dependencias Python...
cd /d "%ROOT_DIR%"
%PYTHON_CMD% -m pip install -r requirements.txt --no-warn-script-location --quiet 2>nul

echo [3/5] Iniciando backend en puerto %BACKEND_PORT%...
set PYTHONPATH=%ROOT_DIR%src
set CSV_PATH=%ROOT_DIR%src\data\steam_rpg_games.csv
set PARAMETERS_PATH=%ROOT_DIR%src\knowledge\parameters.json
set TAGS_PATH=%ROOT_DIR%src\data\tags.csv
:: Matar proceso previo en el puerto si existe
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%BACKEND_PORT% ^| findstr LISTENING') do (
    echo   ^> Cerrando proceso anterior en puerto %BACKEND_PORT%...
    taskkill /f /pid %%a >nul 2>nul
)
timeout /t 1 /nobreak >nul
start "SteamAgent-Backend" cmd /c "%PYTHON_CMD% -m uvicorn api.app:app --reload --env-file .env --host 0.0.0.0 --port %BACKEND_PORT%"
echo   ^> Backend iniciado en ventana separada

timeout /t 3 /nobreak >nul

:: ── Frontend ──────────────────────────────────────────────────────
echo [4/5] Instalando dependencias Flutter...
cd /d "%ROOT_DIR%frontend"
call flutter pub get 2>nul

echo [5/5] Iniciando frontend en Chrome...
echo   ^> API URL: %API_URL%
echo   ^> Cierra la ventana del backend manualmente al terminar
echo.

call flutter run -d chrome --dart-define=API_URL=%API_URL%

echo.
echo ¡Listo!
endlocal
