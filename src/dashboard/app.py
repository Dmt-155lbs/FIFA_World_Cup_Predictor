import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import os
import mlflow

st.set_page_config(
    page_title="Mundial 2026 — Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración y conexión
@st.cache_resource
def get_db_engine():
    # En producción usar variable de entorno: os.getenv("DB_CONNECTION_STRING")
    # Aquí mockeamos la conexión para que funcione la UI básica
    return None

def get_model_versions_from_mlflow():
    # Mock para la UI, en un entorno real leería de mlflow.search_model_versions()
    return ["v20260604_1830 (Production)", "v20260601_1200 (Staging)"]

def get_metric(metric_name, model_version):
    # Mock para la UI
    metrics = {
        "brier_score": 0.2015,
        "roi_flat_stake": 6.8,
        "accuracy": 0.54
    }
    return metrics.get(metric_name, 0)

# Sidebar global
with st.sidebar:
    st.title("🏆 Mundial 2026")
    st.caption("Ensemble: XGBoost + Poisson + Monte Carlo")
    
    model_version = st.selectbox(
        "Versión del Modelo",
        options=get_model_versions_from_mlflow()
    )
    
    st.divider()
    st.metric("Brier Score", f"{get_metric('brier_score', model_version):.4f}")
    st.metric("ROI Backtest", f"{get_metric('roi_flat_stake', model_version):.1f}%")
    st.metric("Simulaciones", "10,000")
    
    st.divider()
    st.info("Desarrollado según la Arquitectura Híbrida.")

st.title("⚽ Predicción del Mundial FIFA 2026")
st.markdown("Bienvenido al Dashboard Analítico. Selecciona una página en el panel lateral para explorar predicciones, simulaciones y explicabilidad del modelo.")
