import glob
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dashboard.data import get_latest_metrics, get_shap_artifact  # noqa: E402

st.set_page_config(page_title="Explicabilidad SHAP", page_icon="🔍", layout="wide")

st.title("🔍 Explicabilidad del Modelo (SHAP)")
st.markdown(
    "Entendiendo el porqué detrás de las predicciones del modelo XGBoost dual "
    "mediante valores SHAP, leídos desde los artefactos de MLflow del último run."
)

meta = get_latest_metrics()
run_id = meta.get("mlflow_run_id")

if not run_id or run_id == "UNKNOWN":
    st.warning(
        "No hay un run de MLflow registrado todavía. Entrena el modelo "
        "(`mundial-cli train`) para generar los artefactos SHAP."
    )
    st.stop()

st.caption(f"Run de MLflow: `{run_id}`")


def _render_images(local_path: str | None, caption_prefix: str) -> bool:
    """Renderiza todas las imágenes PNG encontradas en el artefacto."""
    if not local_path:
        return False
    if os.path.isdir(local_path):
        images = sorted(glob.glob(os.path.join(local_path, "**", "*.png"), recursive=True))
    elif local_path.endswith(".png"):
        images = [local_path]
    else:
        images = []
    for img in images:
        st.image(img, caption=f"{caption_prefix}: {os.path.basename(img)}", use_container_width=True)
    return bool(images)


tab1, tab2 = st.tabs(["Importancia Global (Summary)", "Explicación por Cruce (Waterfall)"])

with tab1:
    st.subheader("Importancia Global de Features")
    st.markdown("Qué variables impactan más en el modelo a nivel macro.")
    path = get_shap_artifact("shap_summary")
    if not _render_images(path, "SHAP Summary"):
        st.info(
            "No se encontraron artefactos `shap_summary` para este run. "
            "Verifica que MLflow esté accesible y que el entrenamiento haya "
            "registrado los plots SHAP."
        )

with tab2:
    st.subheader("Waterfall por Partido de Muestra")
    st.markdown(
        "Descomposición de la contribución de cada feature para un partido "
        "representativo registrado durante el entrenamiento."
    )
    path = get_shap_artifact("shap_waterfall")
    if not _render_images(path, "SHAP Waterfall"):
        st.info(
            "No se encontraron artefactos `shap_waterfall` para este run."
        )
