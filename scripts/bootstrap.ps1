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

Write-Host "==> Esperando a que la base de datos esté lista…" -ForegroundColor Cyan
docker compose exec -T app python -c "import time; from src.utils.db import test_connection; [time.sleep(3) for _ in range(20) if not test_connection()]; print('DB lista' if test_connection() else 'DB no respondió')"

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
