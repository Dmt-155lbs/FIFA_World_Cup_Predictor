import os
import sys

import streamlit as st

# Ajustar path para importar el paquete `dashboard` y `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dashboard.components.heatmap import plot_poisson_heatmap  # noqa: E402
from dashboard.data import list_teams, model_available, predict_match  # noqa: E402

st.set_page_config(page_title="Probabilidades", page_icon="📊", layout="wide")

st.title("📊 Análisis de Probabilidades por Cruce")
st.markdown(
    "Selecciona dos equipos para analizar sus probabilidades en un "
    "enfrentamiento directo usando la Matriz de Poisson Bivariada "
    "(Dixon-Coles), alimentada por el modelo XGBoost dual."
)

teams = list_teams()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Equipo Local (Home)", teams, index=0)
with col2:
    away_team = st.selectbox(
        "Equipo Visitante (Away)", teams, index=1 if len(teams) > 1 else 0
    )

if home_team == away_team:
    st.warning("Selecciona dos equipos distintos.")
    st.stop()

if not model_available():
    st.error(
        "El modelo no está disponible. Entrena el modelo (`mundial-cli train`) "
        "y asegúrate de que el artefacto esté montado para ver predicciones reales."
    )
    st.stop()

pred = predict_match(home_team, away_team)
if pred is None:
    st.error(
        "No se pudo generar la predicción. Revisa que el feature store tenga "
        "datos para ambos equipos y que el modelo esté entrenado."
    )
    st.stop()

st.divider()

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(f"Victoria {home_team}", f"{pred['prob_home']:.1%}")
col_m2.metric("Empate", f"{pred['prob_draw']:.1%}")
col_m3.metric(f"Victoria {away_team}", f"{pred['prob_away']:.1%}")

col_g1, col_g2, col_g3 = st.columns(3)
col_g1.metric(f"Goles esperados {home_team}", f"{pred['expected_home_goals']:.2f}")
col_g2.metric(f"Goles esperados {away_team}", f"{pred['expected_away_goals']:.2f}")
col_g3.metric("Prob. Over 2.5", f"{pred['prob_over_25']:.1%}")

st.divider()

st.subheader("Matriz de Probabilidades Exactas")
fig = plot_poisson_heatmap(pred["score_matrix"], home_team, away_team)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Marcadores más probables")
st.table(
    {
        "Marcador": [f"{h}-{a}" for h, a, _ in pred["most_likely_scores"]],
        "Probabilidad": [f"{p:.1%}" for _, _, p in pred["most_likely_scores"]],
    }
)
