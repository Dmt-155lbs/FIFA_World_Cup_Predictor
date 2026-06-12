# =============================================================================
# bootstrap.ps1 — Arranque "Day 0" del sistema Mundial FIFA 2026 (Windows)
# =============================================================================
# Levanta toda la pila con Docker Compose y ejecuta el pipeline inicial:
#   docker compose up  ->  (el entrypoint del `app` hace init-db + seed)
#   ->  ingest  ->  train  ->  evaluate
#
# Uso:
#   ./scripts/bootstrap.ps1            # pipeline completo
#   ./scripts/bootstrap.ps1 -SkipIngest
# =============================================================================
param(
    [switch]$SkipIngest,
    [switch]$Optimize
)

$ErrorActionPreference = "Stop"

Write-Host "==> Verificando archivo .env" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Write-Host "No existe .env. Copiando desde .env.example…" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Edita .env con tus credenciales reales antes de producción." -ForegroundColor Yellow
}

Write-Host "==> Levantando servicios (docker compose up -d --build)" -ForegroundColor Cyan
docker compose up -d --build

Write-Host "==> Esperando a que la BD esté inicializada y sembrada (db_init + 48 equipos)…" -ForegroundColor Cyan
# No basta con que la conexión responda: el entrypoint corre db_init (schema +
# seed) de forma asíncrona respecto a `up -d`. Esperamos a que DIM_TEAM tenga
# los 48 equipos para no lanzar la ingesta antes de tiempo (evita que todos los
# partidos se descarten por equipos aún no sembrados).
docker compose exec -T app python -c @"
import time
ok = False
for _ in range(40):
    try:
        from src.ingestion.loader import DataLoader
        if DataLoader().get_table_counts().get('DIM_TEAM', 0) >= 48:
            ok = True
            break
    except Exception:
        pass
    time.sleep(3)
print('BD lista (48 equipos sembrados)' if ok else 'TIMEOUT esperando db_init/seed')
"@

if (-not $SkipIngest) {
    Write-Host "==> Ingesta de datos (ingest --source all)" -ForegroundColor Cyan
    docker compose exec -T app python -m src.cli ingest --source all
} else {
    Write-Host "==> Ingesta omitida (-SkipIngest)" -ForegroundColor Yellow
}

Write-Host "==> Entrenamiento del modelo" -ForegroundColor Cyan
if ($Optimize) {
    docker compose exec -T app python -m src.cli train --optimize
} else {
    docker compose exec -T app python -m src.cli train --no-optimize
}

Write-Host "==> Evaluación offline (Walk-Forward + Backtesting)" -ForegroundColor Cyan
docker compose exec -T app python -m src.cli evaluate

Write-Host "==> Listo. Dashboard disponible en http://localhost:8501" -ForegroundColor Green
Write-Host "    MLflow disponible en http://localhost:5000" -ForegroundColor Green
