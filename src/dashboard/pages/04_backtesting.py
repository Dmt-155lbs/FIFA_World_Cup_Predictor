import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dashboard.data import (  # noqa: E402
    db_available,
    get_latest_metrics,
    get_value_bets,
)

st.set_page_config(page_title="Backtesting Financiero", page_icon="📈", layout="wide")

st.title("📈 Backtesting Financiero")
st.markdown(
    "Rentabilidad del modelo evaluada sobre cuotas históricas reales. "
    "Las métricas provienen de la última versión registrada en "
    "`ML_MODEL_VERSION`; las oportunidades se calculan en vivo cruzando el "
    "modelo con las cuotas de `FACT_ODDS`."
)

if not db_available():
    st.error("Sin conexión a la base de datos.")
    st.stop()

metrics = get_latest_metrics()

if not metrics:
    st.warning(
        "No hay métricas registradas. Ejecuta la evaluación offline "
        "(`mundial-cli evaluate`) para poblar `ML_MODEL_VERSION`."
    )
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Brier Score", f"{metrics.get('brier_score', 0.0):.4f}")
    col2.metric("ROI Backtest", f"{metrics.get('roi_backtest', 0.0):.1f}%")
    col3.metric("Log Loss", f"{metrics.get('log_loss', 0.0):.4f}")
    st.caption(
        f"Versión `{metrics.get('version_tag', 'N/A')}` · "
        f"entrenada {metrics.get('trained_at', 'N/A')}"
    )

st.divider()

st.subheader("Próximas Oportunidades (Value Bets)")
st.markdown(
    "Partidos próximos donde el Expected Value (EV) estimado por el modelo "
    "supera el 5% respecto a las cuotas de las casas de apuestas."
)

value_bets = get_value_bets(ev_threshold=0.05)
if value_bets.empty:
    st.info(
        "No hay value bets disponibles. Esto ocurre si no hay cuotas de "
        "partidos futuros en `FACT_ODDS` (ejecuta `mundial-cli ingest "
        "--source odds`) o si el modelo no detecta ventaja > 5%."
    )
else:
    st.dataframe(value_bets, use_container_width=True, hide_index=True)

st.divider()

st.caption(
    "ℹ️ Las curvas de P&L (Flat Stake vs Kelly) se generan durante la "
    "evaluación offline (`run_evaluation`) y se registran como artefactos en "
    "MLflow junto con el ROI agregado mostrado arriba."
)
