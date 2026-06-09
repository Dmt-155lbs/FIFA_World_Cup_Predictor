# 📘 Runbook Operativo — Sistema de Predicción Mundial FIFA 2026

Este manual documenta los procedimientos operativos del sistema en producción (Fase 7), incluyendo la administración del scheduler de reentrenamiento, visualización de logs y manejo de alertas.

## 1. Arquitectura de Despliegue
El sistema se orquesta mediante `docker-compose` con 4 servicios principales:
- `sqlserver`: Base de datos transaccional con el esquema estrella.
- `mlflow`: Registro de experimentos y tracking de modelos.
- `app`: Daemon principal que ejecuta `src.scheduler`. Se encarga de ingestar y reentrenar.
- `streamlit`: Dashboard interactivo (expuesto en puerto 8501).

## 2. Gestión de Servicios (Docker Compose)

### 2.1 Levantar el sistema
```bash
docker-compose up -d --build
```
> Esto iniciará todos los servicios. El contenedor `app` ahora levantará automáticamente el `BackgroundScheduler` y quedará corriendo en segundo plano esperando el trigger de sus tareas (cron).

### 2.2 Ver logs del Scheduler (App)
Para visualizar cuándo se ejecuta la ingesta o el reentrenamiento:
```bash
docker logs -f mundial-app
```
> Busca mensajes como `Executing Ingestion Job` o `Executing Retraining Job`.

### 2.3 Detener el sistema
```bash
docker-compose down
```

## 3. Scheduler y Trabajos (Jobs)

El contenedor `app` administra los siguientes cron jobs (`src/scheduler.py`):
1. **Ingesta Post-Partido:** Cada 4 horas. (Scraping de FBRef y Odds API).
2. **Reentrenamiento del Modelo:** Diariamente a las 02:00 UTC. (Pipeline de XGBoost y recalibración de Dixon-Coles).
3. **Re-Simulación Monte Carlo:** Diariamente a las 06:00 UTC.

### Ejecución Manual de Trabajos
Si necesitas forzar la ejecución de un trabajo fuera de su horario (por ejemplo, para actualizar predicciones de inmediato tras un partido importante), puedes lanzar un proceso efímero usando el mismo contenedor:

**Forzar Ingesta:**
```bash
docker-compose exec app python -c "from src.scheduler import run_ingestion_job; run_ingestion_job()"
```

**Forzar Reentrenamiento:**
```bash
docker-compose exec app python -c "from src.scheduler import run_retrain_job; run_retrain_job()"
```

## 4. Alertas y Monitoreo

El sistema está equipado con un módulo de alertas (`src/utils/alerts.py`) que notifica fallos críticos.

### Configuración del Webhook
Para recibir alertas en Slack o Discord, define la variable de entorno `WEBHOOK_URL` en tu archivo `.env`:
```env
WEBHOOK_URL=https://hooks.slack.com/services/T0000/B0000/XXXX
```
> Si esta variable está vacía o no existe, las alertas se registran únicamente en los logs del contenedor `app` (`docker logs mundial-app`).

### Eventos Alertas
- **Nivel CRÍTICO:** Fallos de conexión en la ingesta, excepciones no controladas, o degradación del modelo (`Brier Score > 0.23` tras reentrenamiento).
- **Nivel INFO:** Reentrenamiento exitoso con mejoras en las métricas.

## 5. Troubleshooting (Solución de problemas comunes)

| Problema | Causa Posible | Solución |
|----------|---------------|----------|
| `ConnectionRefusedError` al iniciar `app` | SQL Server aún no está listo | El contenedor `app` tiene políticas de reintento, pero si falla definitivamente, reinicia el contenedor: `docker restart mundial-app` |
| Scraping falla silenciosamente | Bloqueo por Rate Limit o caché corrupta | El scheduler usa `fetch_with_retry`. Si falla tras los reintentos, lanzará una alerta. Puedes intentar limpiar la caché: `rm -rf data/cache/*` y ejecutar la ingesta manual. |
| Degradación de modelo persistente | Nuevos datos afectan la distribución (Drift) | Revisa la página de SHAP en el dashboard para identificar si alguna característica nueva (ej. lesiones graves afectando el Elo) está distorsionando los lambdas. |
