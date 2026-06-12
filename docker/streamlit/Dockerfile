# =============================================================================
# Dockerfile — Dashboard Streamlit
# =============================================================================
# Imagen ligera para ejecutar el dashboard interactivo de predicciones.
# Incluye Streamlit, Plotly y dependencias de visualización.
# =============================================================================

FROM python:3.11-slim

# ---- Metadatos de la imagen ----
LABEL maintainer="Mundial 2026 Team"
LABEL description="Dashboard Streamlit — Visualización de predicciones"

# ---- Variables de entorno ----
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- Instalar dependencias del sistema ----
# ODBC Driver 18 para conectarse a SQL Server desde el dashboard
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
# Incluye las dependencias de inferencia (xgboost, shap, optuna, yaml…) porque
# el dashboard ejecuta el pipeline real en vivo (predicción + Monte Carlo).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        streamlit>=1.36 \
        plotly \
        matplotlib \
        seaborn \
        pandas \
        numpy \
        scipy \
        scikit-learn \
        sqlalchemy>=2.0 \
        pyodbc \
        mlflow>=2.14 \
        pydantic>=2.0 \
        pydantic-settings \
        xgboost>=2.1 \
        joblib \
        optuna \
        shap>=0.45 \
        structlog \
        tenacity \
        pyyaml \
        requests

# ---- Dependencia runtime del driver ODBC (Kerberos GSSAPI) ----
# Necesaria para que msodbcsql18 cargue (la purga --auto-remove la elimina).
RUN apt-get update && apt-get install -y --no-install-recommends libgssapi-krb5-2 \
    && rm -rf /var/lib/apt/lists/*

# ---- Directorio de trabajo ----
WORKDIR /app

# ---- Configuración de Streamlit ----
# Desactivar la recopilación de telemetría y configurar el servidor
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# ---- Exponer puerto de Streamlit ----
EXPOSE 8501

# ---- Comando de inicio ----
# Iniciar Streamlit apuntando al módulo principal del dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
