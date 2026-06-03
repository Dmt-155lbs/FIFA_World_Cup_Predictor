# =============================================================================
# Dockerfile — Servicio principal de predicción (App)
# =============================================================================
# Imagen base ligera con Python 3.11.
# Incluye ODBC Driver 18 para conectividad con SQL Server.
# XGBoost se ejecuta solo en CPU (tree_method='hist').
# =============================================================================

FROM python:3.11-slim

# ---- Metadatos de la imagen ----
LABEL maintainer="Mundial 2026 Team"
LABEL description="Servicio principal de predicción — Mundial FIFA 2026"

# ---- Variables de entorno ----
# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
# Asegurar que la salida de Python no se almacene en buffer
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---- Instalar dependencias del sistema ----
# Incluye ODBC Driver 18 para SQL Server, herramientas de compilación y gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    apt-transport-https \
    build-essential \
    gcc \
    g++ \
    unixodbc-dev \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get purge -y --auto-remove curl gnupg2 apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# ---- Directorio de trabajo ----
WORKDIR /app

# ---- Copiar e instalar dependencias Python ----
# Se copia primero pyproject.toml para aprovechar la caché de capas de Docker
COPY ../../pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]" \
    || pip install --no-cache-dir \
        xgboost>=2.1 \
        scikit-learn \
        scipy \
        numpy \
        numba \
        pandas \
        pydantic>=2.0 \
        pydantic-settings \
        sqlalchemy>=2.0 \
        pyodbc \
        mlflow>=2.14 \
        shap>=0.45 \
        optuna \
        streamlit>=1.36 \
        tenacity \
        structlog \
        matplotlib \
        seaborn \
        plotly \
        requests \
        pyyaml \
        apscheduler \
        properscoring \
        soccerdata>=1.3 \
        pytest \
        pytest-cov

# ---- Comando por defecto ----
# Mantener el contenedor en ejecución (los pipelines se ejecutan bajo demanda)
CMD ["tail", "-f", "/dev/null"]
