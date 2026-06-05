import streamlit as st
import os

st.set_page_config(page_title="Explicabilidad SHAP", page_icon="🔍", layout="wide")

st.title("🔍 Explicabilidad del Modelo (SHAP)")
st.markdown("Entendiendo el porqué detrás de las predicciones del modelo XGBoost dual mediante valores SHAP (SHapley Additive exPlanations).")

tab1, tab2 = st.tabs(["Importancia Global (Summary)", "Explicación por Cruce (Force Plot)"])

with tab1:
    st.subheader("Importancia Global de Features")
    st.markdown("Muestra qué variables impactan más en el modelo a nivel macro.")
    
    # Placeholder for image
    st.info("Aquí se renderiza `shap_summary.png` desde MLflow Artifacts.")
    st.code("mlflow.artifacts.download_artifacts('runs:/<run_id>/shap_summary.png')")

with tab2:
    st.subheader("Force Plot por Partido")
    st.markdown("Análisis detallado de un cruce específico.")
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Local", ["Argentina", "Brasil", "Francia", "Inglaterra", "España"], key="shap_home")
    with col2:
        away_team = st.selectbox("Visitante", ["Brasil", "Francia", "Inglaterra", "España", "Argentina"], key="shap_away")
        
    st.info(f"Aquí se renderiza el Force Plot interactivo para {home_team} vs {away_team}")
    st.markdown("""
    > **Ejemplo de interpretación:** El Elo de Argentina empuja la predicción de $\lambda_{home}$ hacia la derecha (+0.3 goles), mientras que la fatiga acumulada la empuja ligeramente a la izquierda (-0.05 goles).
    """)
