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

echo "==> Esperando a que la BD esté inicializada y sembrada (db_init + 48 equipos)…"
# No basta con la conexión: el entrypoint corre db_init (schema + seed) de forma
# asíncrona respecto a `up -d`. Esperamos a que DIM_TEAM tenga los 48 equipos.
docker compose exec -T app python -c '
import time
ok = False
for _ in range(40):
    try:
        from src.ingestion.loader import DataLoader
        if DataLoader().get_table_counts().get("DIM_TEAM", 0) >= 48:
            ok = True
            break
    except Exception:
        pass
    time.sleep(3)
print("BD lista (48 equipos sembrados)" if ok else "TIMEOUT esperando db_init/seed")
'

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
