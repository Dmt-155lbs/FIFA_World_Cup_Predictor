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

st.subheader("🎯 Apuestas de Valor Actuales")
st.markdown(
    "Para los **partidos que se vienen** del Mundial, cruzamos la probabilidad "
    "del modelo contra las cuotas reales de las casas de apuestas (feed en vivo "
    "de The Odds API). Se marca *value bet* cada selección cuyo **Expected "
    "Value** — `EV = Prob. Modelo × Cuota − 1` — supera el umbral elegido: ahí "
    "el modelo estima más probabilidad de la que el mercado está pagando."
)

ev_pct = st.slider(
    "Umbral mínimo de EV", min_value=0, max_value=20, value=5, step=1,
    format="%d%%",
    help="Sólo se listan apuestas cuyo Expected Value supera este porcentaje.",
)

value_bets = get_value_bets(ev_threshold=ev_pct / 100.0)

c1, c2 = st.columns([1, 3])
c1.metric("Value bets detectadas", len(value_bets))
c2.caption(
    "Comparativa por selección: **Prob. Modelo** (lo que cree el modelo) vs "
    "**Prob. Casa** (probabilidad implícita del mercado, ya descontado el "
    "margen de la casa). Si Modelo > Casa lo suficiente, hay valor."
)

if value_bets.empty:
    st.info(
        "No hay value bets por encima del umbral. Esto ocurre si todavía no "
        "hay cuotas de partidos futuros en `FACT_ODDS` (ejecuta `mundial-cli "
        "ingest --source odds` para traer el feed en vivo) o si el modelo no "
        "detecta ventaja sobre el mercado en los fixtures actuales."
    )
else:
    st.dataframe(
        value_bets,
        use_container_width=True,
        hide_index=True,
        column_config={
            "EV": st.column_config.TextColumn(
                "EV", help="Expected Value = Prob. Modelo × Cuota − 1"
            ),
            "Cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
        },
    )
    st.caption(
        "⚠️ **Fiabilidad** = `Baja` cuando (1) algún equipo carece de historial "
        "—ya resuelto: los 48 lo tienen tras cargar los 12 que faltaban— o (2) el "
        "EV es implausible (>35%, techo de un mercado líquido) o la `Prob. Modelo` "
        "dispara respecto a la `Prob. Casa` (≥2× y ≥10 pp): ahí el modelo Poisson "
        "**sobre-estima al underdog** (sesgo estructural, no de datos) y el EV no "
        "es ventaja real. Confía en las filas `Alta` (se muestran primero). El EV "
        "positivo es esperanza matemática sobre la cuota, no una garantía."
    )

st.divider()

st.caption(
    "ℹ️ Las métricas de arriba (Brier / Log Loss) provienen de la evaluación "
    "walk-forward sobre partidos **ya jugados**. El ROI del backtest histórico "
    "requiere cuotas históricas (plan de pago de The Odds API); esta sección de "
    "Value Bets es el cruce modelo-vs-mercado sobre los partidos **futuros**, la "
    "vía para la que está construido el predictor."
)
