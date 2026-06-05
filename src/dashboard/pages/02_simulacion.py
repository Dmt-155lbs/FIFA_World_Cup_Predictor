import streamlit as st
import sys
import os

# Ajustar path para importación de componentes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dashboard.components.bracket_viz import plot_bracket, plot_champions_distribution

st.set_page_config(page_title="Simulación Monte Carlo", page_icon="🎯", layout="wide")

st.title("🎯 Simulación Estructural (Monte Carlo)")
st.markdown("Basado en 10,000 iteraciones del torneo completo usando las predicciones de Poisson y considerando el formato de 48 equipos.")

st.subheader("Distribución de Campeones (Top 10)")
# Mock data
mock_champions = {
    "Argentina": 0.185,
    "Francia": 0.162,
    "Brasil": 0.141,
    "Inglaterra": 0.115,
    "España": 0.098,
    "Alemania": 0.082,
    "Portugal": 0.065,
    "Holanda": 0.045,
    "Italia": 0.032,
    "Bélgica": 0.021
}

fig_champs = plot_champions_distribution(mock_champions)
st.plotly_chart(fig_champs, use_container_width=True)

st.divider()

st.subheader("Simulador de Bracket interactivo")
st.markdown("Visualiza el camino proyectado para la fase eliminatoria.")

sim_id = st.slider("Seleccionar Iteración de Simulación", 1, 10000, 1)

fig_bracket = plot_bracket(None)
st.plotly_chart(fig_bracket, use_container_width=True)
