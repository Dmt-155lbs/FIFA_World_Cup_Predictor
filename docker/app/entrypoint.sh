#!/usr/bin/env bash
# =============================================================================
# Entrypoint del servicio `app`.
# SQL Server no auto-ejecuta los scripts de db/init, por lo que inicializamos
# el esquema y las dimensiones de forma programática al arrancar (idempotente),
# y luego cedemos el control al proceso principal (scheduler por defecto).
# =============================================================================
set -euo pipefail

echo "[entrypoint] Inicializando base de datos (schema + seed)…"
if python -m src.ingestion.db_init; then
    echo "[entrypoint] Base de datos inicializada correctamente."
else
    echo "[entrypoint] Aviso: db_init no completó (¿ya inicializada?). Continuando."
fi

echo "[entrypoint] Lanzando proceso principal: $*"
exec "$@"
