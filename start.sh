#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Colores ─────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"

# ── Ayuda ───────────────────────────────────────────────────────────
usage() {
    echo -e "${BOLD}Uso:${NC} ./start.sh [opciones]"
    echo ""
    echo "Opciones:"
    echo "  --api-url URL   URL del backend (default: http://localhost:8000)"
    echo "  --port PUERTO   Puerto del backend (default: 8000)"
    echo "  --help          Muestra esta ayuda"
    exit 0
}

API_URL="http://localhost:8000"
BACKEND_PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-url)    API_URL="$2";    shift 2 ;;
        --port)       BACKEND_PORT="$2"; shift 2 ;;
        --help|-h)    usage ;;
        *)            echo -e "${RED}Argumento desconocido: $1${NC}"; usage ;;
    esac
done

echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     SteamAgent Recommender — Launcher     ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── .env ────────────────────────────────────────────────────────────
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo -e "${RED}Error: No existe el archivo .env${NC}"
    echo -e "  Copia .env.example a .env y agrega tu Steam API Key:${NC}"
    echo -e "  cp .env.example .env${NC}"
    echo -e "  Luego edita .env y pon tu clave de: https://steamcommunity.com/dev/apikey${NC}"
    exit 1
fi

# ── Requisitos ──────────────────────────────────────────────────────
echo -e "${YELLOW}[1/5] Verificando requisitos...${NC}"

PYTHON=""
if command -v py &>/dev/null; then
    PYTHON="py -3"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    if python -c "print('ok')" &>/dev/null; then
        PYTHON="python"
    fi
fi

if [ -z "$PYTHON" ]; then
    echo -e "${RED}Error: Python no encontrado.${NC}"
    echo -e "  Instálalo desde https://www.python.org/ (marca 'Add Python to PATH')${NC}"
    echo -e "  y deshabilita los alias de Microsoft Store en:${NC}"
    echo -e "  Configuración > Aplicaciones > Alias de ejecución de aplicaciones${NC}"
    exit 1
fi

if ! command -v flutter &>/dev/null; then
    echo -e "${RED}Error: Flutter no encontrado. Instálalo desde https://flutter.dev/${NC}"
    exit 1
fi

# ── Backend ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/5] Instalando dependencias Python...${NC}"
cd "$ROOT_DIR"
$PYTHON -m pip install -r requirements.txt --no-warn-script-location --quiet 2>/dev/null

echo -e "${YELLOW}[3/5] Iniciando backend en puerto ${BACKEND_PORT}...${NC}"

# Matar proceso previo en el puerto si existe
OLD_PID=$(lsof -ti tcp:"$BACKEND_PORT" 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo -e "  → Cerrando proceso anterior en puerto $BACKEND_PORT (PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

PYTHONPATH="$ROOT_DIR/src" \
CSV_PATH="$ROOT_DIR/src/data/steam_rpg_games.csv" \
PARAMETERS_PATH="$ROOT_DIR/src/knowledge/parameters.json" \
TAGS_PATH="$ROOT_DIR/src/data/tags.csv" \
$PYTHON -m uvicorn api.app:app \
    --reload \
    --reload-exclude ".env" \
    --env-file "$ROOT_DIR/.env" \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" &
BACKEND_PID=$!
echo -e "${GREEN}  → Backend PID: $BACKEND_PID${NC}"

sleep 3

# ── Frontend ────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/5] Instalando dependencias Flutter...${NC}"
cd "$ROOT_DIR/frontend"
flutter pub get 2>/dev/null

echo -e "${YELLOW}[5/5] Iniciando frontend en Chrome...${NC}"
echo -e "${CYAN}  → API URL: ${API_URL}${NC}"
echo -e "${CYAN}  → Backend PID: $BACKEND_PID${NC}"
echo -e "${GREEN}  → Presiona Ctrl+C para detener todo${NC}"
echo ""

flutter run -d chrome --dart-define=API_URL="$API_URL"

# ── Limpieza ────────────────────────────────────────────────────────
echo -e "${YELLOW}Deteniendo backend (PID $BACKEND_PID)...${NC}"
kill "$BACKEND_PID" 2>/dev/null || true
echo -e "${GREEN}¡Listo!${NC}"
