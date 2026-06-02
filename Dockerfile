FROM python:3.11-slim

# Instalar SWI-Prolog y dependencias del sistema
RUN apt-get update && apt-get install -y \
    swi-prolog \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/

# Variables de entorno por defecto
ENV PYTHONPATH=src/
ENV CSV_PATH=src/data/steam_rpg_games.csv
ENV PARAMETERS_PATH=src/knowledge/parameters.jsonPARAMETERS_PATH=src/knowledge/parameters.json

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]