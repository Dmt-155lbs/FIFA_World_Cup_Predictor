#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — Arranque "Day 0" del sistema Mundial FIFA 2026 (Linux/macOS)
# =============================================================================
# Levanta toda la pila con Docker Compose y ejecuta el pipeline inicial:
#   docker compose up  ->  (el entrypoint del `app` hace init-db + seed)
#   ->  ingest  ->  train  ->  evaluate
#
# Uso:
#   ./scripts/bootstrap.sh                 # pipeline completo (sin Optuna)
#   SKIP_INGEST=1 ./scripts/bootstrap.sh   # omitir ingesta
#   OPTIMIZE=1 ./scripts/bootstrap.sh      # entrenar con Optuna
# =============================================================================
set -euo pipefail

echo "==> Verificando archivo .env"
if [ ! -f .env ]; then
    echo "No existe .env. Copiando desde .env.example…"
    cp .env.example .env
    echo "Edita .env con tus credenciales reales antes de producción."
fi

echo "==> Levantando servicios (docker compose up -d --build)"
docker compose up -d --build

echo "==> Esperando a que la base de datos esté lista…"
docker compose exec -T app python -c "import time; from src.utils.db import test_connection; [time.sleep(3) for _ in range(20) if not test_connection()]; print('DB lista' if test_connection() else 'DB no respondio')"

if [ "${SKIP_INGEST:-0}" != "1" ]; then
    echo "==> Ingesta de datos (ingest --source all)"
    docker compose exec -T app python -m src.cli ingest --source all
else
    echo "==> Ingesta omitida (SKIP_INGEST=1)"
fi

echo "==> Entrenamiento del modelo"
if [ "${OPTIMIZE:-0}" = "1" ]; then
    docker compose exec -T app python -m src.cli train --optimize
else
    docker compose exec -T app python -m src.cli train --no-optimize
fi

echo "==> Evaluación offline (Walk-Forward + Backtesting)"
docker compose exec -T app python -m src.cli evaluate

echo "==> Listo. Dashboard disponible en http://localhost:8501"
echo "    MLflow disponible en http://localhost:5000"
