import os
import sys

import streamlit as st

# Permitir importar el paquete `dashboard` y `src` cuando Streamlit ejecuta
# este archivo como script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.data import (  # noqa: E402
    db_available,
    get_latest_metrics,
    get_model_versions,
    model_available,
)

st.set_page_config(
    page_title="Mundial 2026 — Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar global — estado del sistema, versión del modelo y métricas reales
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏆 Mundial 2026")
    st.caption("Ensemble: XGBoost + Poisson + Monte Carlo")

    # Indicadores de conectividad
    db_ok = db_available()
    model_ok = model_available()
    st.write(
        f"**Base de datos:** {'🟢 Conectada' if db_ok else '🔴 Sin conexión'}"
    )
    st.write(
        f"**Modelo:** {'🟢 Cargado' if model_ok else '🟠 No disponible'}"
    )

    st.divider()

    versions_df = get_model_versions()
    if not versions_df.empty:
        version_labels = versions_df["version_tag"].tolist()
        selected = st.selectbox("Versión del Modelo", options=version_labels)
        row = versions_df[versions_df["version_tag"] == selected].iloc[0]

        st.metric("Brier Score", f"{row['brier_score']:.4f}")
        st.metric("ROI Backtest", f"{row['roi_backtest']:.1f}%")
        st.metric("Log Loss", f"{row['log_loss']:.4f}")
        st.caption(f"Entrenado: {row['trained_at']}")
        st.caption(f"MLflow run: `{row['mlflow_run_id']}`")
    else:
        st.warning(
            "No hay versiones de modelo registradas en la BD. "
            "Ejecuta `mundial-cli train` para generar la primera."
        )

    st.divider()
    st.info("Desarrollado según la Arquitectura Híbrida de 3 capas.")

# ---------------------------------------------------------------------------
# Página principal
# ---------------------------------------------------------------------------
st.title("⚽ Predicción del Mundial FIFA 2026")
st.markdown(
    "Bienvenido al Dashboard Analítico. Selecciona una página en el panel "
    "lateral para explorar predicciones, simulaciones, explicabilidad y "
    "backtesting financiero del modelo."
)

if not db_available():
    st.error(
        "⚠️ Sin conexión a la base de datos. Verifica que el contenedor "
        "`sqlserver` esté activo y que las variables `DB_*` estén definidas."
    )
elif not model_available():
    st.warning(
        "El modelo aún no está entrenado o el artefacto no es accesible. "
        "Las páginas que dependen de inferencia (Probabilidades, Simulación) "
        "mostrarán un aviso hasta que ejecutes el entrenamiento "
        "(`mundial-cli train`) y el volumen de modelos esté montado."
    )
else:
    latest = get_latest_metrics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Brier Score (última versión)", f"{latest.get('brier_score', 0.0):.4f}")
    col2.metric("ROI Backtest", f"{latest.get('roi_backtest', 0.0):.1f}%")
    col3.metric("Log Loss", f"{latest.get('log_loss', 0.0):.4f}")
    st.success("Sistema operativo: BD conectada y modelo cargado. ✅")
