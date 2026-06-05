# run_mlflow.ps1
# Script de conveniencia para iniciar el servidor de tracking de MLflow en local.

Write-Host "Iniciando servidor de UI de MLflow en http://localhost:5000..." -ForegroundColor Green
Write-Host "Presiona Ctrl+C para detener." -ForegroundColor Yellow

# Ejecutar MLflow sirviendo en el puerto 5000
mlflow ui --host 127.0.0.1 --port 5000
