import streamlit as st
import numpy as np
import sys
import os

# Ajustar path para importación de componentes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dashboard.components.heatmap import plot_poisson_heatmap

st.set_page_config(page_title="Probabilidades", page_icon="📊", layout="wide")

st.title("📊 Análisis de Probabilidades por Cruce")
st.markdown("Selecciona dos equipos para analizar sus probabilidades en un enfrentamiento directo usando la Matriz de Poisson Bivariada (Dixon-Coles).")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Equipo Local (Home)", ["Argentina", "Brasil", "Francia", "Inglaterra", "España"])
with col2:
    away_team = st.selectbox("Equipo Visitante (Away)", ["Brasil", "Francia", "Inglaterra", "España", "Argentina"])

# Mock de la matriz de Poisson para la demostración
# En producción, esto vendría de `pipeline.predict_match()`
@st.cache_data
def get_mock_poisson_matrix():
    matrix = np.zeros((6, 6))
    lambda_h = 1.5
    lambda_a = 1.2
    for i in range(6):
        for j in range(6):
            # Aproximación simple para mock
            matrix[i, j] = ((lambda_h**i * np.exp(-lambda_h)) / np.math.factorial(i)) * \
                           ((lambda_a**j * np.exp(-lambda_a)) / np.math.factorial(j))
    # Normalizar
    matrix /= matrix.sum()
    return matrix

matrix = get_mock_poisson_matrix()

# Calcular agregados
prob_home = np.tril(matrix, -1).sum()
prob_draw = np.diag(matrix).sum()
prob_away = np.triu(matrix, 1).sum()

st.divider()

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(f"Victoria {home_team}", f"{prob_home:.1%}")
col_m2.metric("Empate", f"{prob_draw:.1%}")
col_m3.metric(f"Victoria {away_team}", f"{prob_away:.1%}")

st.divider()

st.subheader("Matriz de Probabilidades Exactas")
fig = plot_poisson_heatmap(matrix, home_team, away_team)
st.plotly_chart(fig, use_container_width=True)
