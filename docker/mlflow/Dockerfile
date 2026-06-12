# =============================================================================
# Dockerfile — Servidor MLflow
# =============================================================================
# Imagen ligera para ejecutar el servidor de seguimiento de MLflow.
# Backend store: SQL Server (base de datos 'mlflow')
# Artifact store: directorio local montado como volumen
# =============================================================================

FROM python:3.11-slim

# ---- Metadatos de la imagen ----
LABEL maintainer="Mundial 2026 Team"
LABEL description="Servidor MLflow — Seguimiento de experimentos"

# ---- Variables de entorno ----
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- Instalar dependencias del sistema ----
# Se necesita ODBC Driver 18 para la conexión con SQL Server
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    apt-transport-https \
    unixodbc-dev \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get purge -y --auto-remove curl gnupg2 apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# ---- Instalar dependencias Python ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        mlflow>=2.14 \
        pyodbc

# ---- Dependencia de runtime del driver ODBC ----
# El paso de `apt-get purge --auto-remove` de arriba elimina libgssapi-krb5-2,
# que el driver msodbcsql18 necesita en runtime. Sin esta librería, pyodbc
# falla al cargar el driver con un engañoso "file not found"
# (libmsodbcsql-18.6.so.2.1) y el servidor MLflow no puede conectar a su
# backend SQL Server. Se reinstala explícitamente para que no sea removible.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgssapi-krb5-2 \
    && rm -rf /var/lib/apt/lists/*

# ---- Crear directorio para artefactos ----
RUN mkdir -p /mlflow/artifacts

# ---- Exponer puerto del servidor MLflow ----
EXPOSE 5000

# ---- Comando de inicio ----
# Iniciar el servidor MLflow con backend en SQL Server y artefactos locales
# La contraseña de SA se inyecta como variable de entorno desde docker-compose
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "mssql+pyodbc://sa:${DB_SA_PASSWORD}@sqlserver:1433/mlflow?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes" \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts
