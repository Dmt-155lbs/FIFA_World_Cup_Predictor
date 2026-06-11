import os
import sys

import streamlit as st

# Ajustar path para importar el paquete `dashboard` y `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dashboard.components.bracket_viz import (  # noqa: E402
    plot_bracket,
    plot_champions_distribution,
)
from dashboard.data import model_available, run_simulation  # noqa: E402

st.set_page_config(page_title="Simulación Monte Carlo", page_icon="🎯", layout="wide")

st.title("🎯 Simulación Estructural (Monte Carlo)")
st.markdown(
    "Simulación del torneo completo (48 equipos) usando las fuerzas base "
    "derivadas del modelo (`team_lambda_builder`) y la matriz de Poisson."
)

if not model_available():
    st.error(
        "El modelo no está disponible. Entrena el modelo (`mundial-cli train`) "
        "para ejecutar simulaciones reales."
    )
    st.stop()

n_sims = st.select_slider(
    "Número de simulaciones",
    options=[1000, 2500, 5000, 10000, 25000],
    value=10000,
)

results = run_simulation(n_sims)
if results is None:
    st.error(
        "No se pudo ejecutar la simulación. Revisa el feature store y el modelo."
    )
    st.stop()

st.caption(f"Basado en {results['n_simulations']:,} iteraciones del torneo.")

st.subheader("Distribución de Campeones (Top 10)")
fig_champs = plot_champions_distribution(results["champion_probs"])
st.plotly_chart(fig_champs, use_container_width=True)

st.divider()

st.subheader("Probabilidad de avance por ronda")
fig_bracket = plot_bracket(results["round_advance_probs"])
st.plotly_chart(fig_bracket, use_container_width=True)

st.divider()

with st.expander("Ver fuerzas base estimadas (team_lambdas)"):
    st.markdown("`(λ ataque, λ defensa)` por equipo — derivadas del modelo entrenado.")
    lambdas = results["team_lambdas"]
    st.table(
        {
            "Equipo": list(lambdas.keys()),
            "λ Ataque": [round(v[0], 3) for v in lambdas.values()],
            "λ Defensa": [round(v[1], 3) for v in lambdas.values()],
        }
    )
