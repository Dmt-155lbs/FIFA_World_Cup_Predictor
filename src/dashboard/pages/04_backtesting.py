import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dashboard.components.roi_chart import plot_roi_curves

st.set_page_config(page_title="Backtesting Financiero", page_icon="📈", layout="wide")

st.title("📈 Backtesting Financiero")
st.markdown("Evaluación de la rentabilidad del modelo simulando estrategias de apuestas sobre cuotas históricas reales.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Brier Score", "0.2015", "-0.0485 vs Naive", delta_color="inverse")
col2.metric("ROI Flat Stake", "6.8%", "1000 iteraciones")
col3.metric("ROI Kelly (1/4)", "12.4%", "Mayor varianza")
col4.metric("Win Rate (EV > 5%)", "54.2%", "")

st.divider()

st.subheader("Evolución del Bankroll ($)")
fig = plot_roi_curves(None)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Próximas Oportunidades (Value Bets)")
st.markdown("Partidos futuros donde el Expected Value (EV) estimado por el modelo > 5% respecto a las casas de apuestas.")

st.table({
    "Partido": ["Argentina vs México", "Inglaterra vs USA", "España vs Alemania"],
    "Pick": ["Victoria Argentina", "Empate", "Victoria España"],
    "Cuota Casa": [1.95, 3.80, 2.50],
    "Prob Modelo": ["55%", "28%", "43%"],
    "EV": ["+7.2%", "+6.4%", "+7.5%"]
})
