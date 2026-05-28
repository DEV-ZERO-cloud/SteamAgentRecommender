FROM python:3.11-slim

# Instalar SWI-Prolog y dependencias del sistema
RUN apt-get update && apt-get install -y \
    swi-prolog \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# --- Layer 1: torch CPU-only (pesado, se cachea por separado) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu

# --- Layer 2: resto de dependencias ---
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/

# Variables de entorno por defecto (overridable via docker-compose)
ENV PYTHONPATH=src/
ENV CSV_PATH=src/data/steam_rpg_games.csv
ENV PARAMETERS_PATH=src/knowledge/parameters.json
ENV EMBEDDINGS_CACHE_PATH=src/embeddings/embeddings_cache.npy

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]