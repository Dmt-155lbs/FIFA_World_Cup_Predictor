"""
Análisis de explicabilidad de modelos usando SHAP (SHapley Additive exPlanations).

Permite interpretar el impacto de las variables en los modelos XGBoost local y
visitante mediante TreeExplainer y visualizaciones estándar.

Autor: Mundial 2026 Team
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import structlog
from xgboost import XGBRegressor

logger = structlog.get_logger(__name__)


class SHAPAnalyzer:
    """
    Explicabilidad con TreeExplainer para los modelos XGBoost del ensemble.

    Calcula los valores SHAP para interpretar qué features están guiando
    las predicciones de los goles esperados (lambdas).
    """

    def __init__(
        self,
        model_home: XGBRegressor,
        model_away: XGBRegressor,
        feature_names: list[str],
    ) -> None:
        """
        Inicializa los explainers de SHAP para ambos modelos.

        Args:
            model_home: Modelo XGBoost entrenado para predecir goles locales.
            model_away: Modelo XGBoost entrenado para predecir goles visitantes.
            feature_names: Lista ordenada con los nombres de los features.
        """
        logger.info("Inicializando SHAP TreeExplainers...")
        self.explainer_home = shap.TreeExplainer(model_home)
        self.explainer_away = shap.TreeExplainer(model_away)
        self.feature_names = feature_names

    def compute_shap_values(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los valores SHAP para un conjunto de features.

        Args:
            X: DataFrame de features a interpretar.

        Returns:
            Tupla (shap_home, shap_away) con los valores SHAP en numpy arrays.
        """
        logger.debug(f"Calculando SHAP values para {len(X)} observaciones")
        # check_additivity=False es OBLIGATORIO: los modelos usan el objetivo
        # 'count:poisson' (link logarítmico), por lo que model.predict() devuelve
        # exp(margin) mientras que TreeExplainer explica el margin crudo. La
        # comprobación de aditividad por defecto compara ambos y lanza
        # "Additivity check failed", abortando todo el análisis SHAP.
        shap_home = self.explainer_home.shap_values(X, check_additivity=False)
        shap_away = self.explainer_away.shap_values(X, check_additivity=False)
        return shap_home, shap_away

    def plot_summary(
        self,
        X: pd.DataFrame,
        save_dir: str = "./artifacts",
        prefix: str = "",
    ) -> list[str]:
        """
        Genera gráficos de resumen (Summary Plot) para la importancia global.

        El gráfico 'dot' muestra la distribución del impacto de cada feature
        en la predicción del modelo. Se generan dos gráficos: local y visitante.

        Args:
            X: DataFrame con las features de prueba o validación.
            save_dir: Directorio base donde guardar las imágenes.
            prefix: Prefijo opcional para los nombres de archivo.

        Returns:
            Lista con las rutas absolutas donde se guardaron las imágenes.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        shap_home, shap_away = self.compute_shap_values(X)

        saved_files = []

        # --- Gráfico Local ---
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_home,
            X,
            feature_names=self.feature_names,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Summary Plot - Modelo Local")
        plt.tight_layout()
        home_path = os.path.join(save_dir, f"{prefix}shap_summary_home.png")
        plt.savefig(home_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(os.path.abspath(home_path))

        # --- Gráfico Visitante ---
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_away,
            X,
            feature_names=self.feature_names,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Summary Plot - Modelo Visitante")
        plt.tight_layout()
        away_path = os.path.join(save_dir, f"{prefix}shap_summary_away.png")
        plt.savefig(away_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(os.path.abspath(away_path))

        logger.info(f"Summary plots generados: {saved_files}")
        return saved_files

    def explain_match(
        self,
        match_features: pd.DataFrame,
        team_home: str = "Local",
        team_away: str = "Visitante",
        save_dir: str = "./artifacts",
    ) -> list[str]:
        """
        Genera un Force Plot estático para explicar una predicción individual.

        En lugar de usar JS para interactividad, esto genera imágenes .png
        que pueden subirse a MLflow como artefactos estáticos (waterfall/force).
        Utilizamos waterfall plot ya que matplotlib force plots se deprecated.

        Args:
            match_features: DataFrame con 1 fila correspondiente al partido.
            team_home: Nombre del equipo local.
            team_away: Nombre del equipo visitante.
            save_dir: Directorio donde guardar las imágenes.

        Returns:
            Lista de rutas donde se guardaron las imágenes.
        """
        if len(match_features) != 1:
            raise ValueError("match_features debe contener exactamente 1 fila.")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        saved_files = []
        match_idx = match_features.index[0]

        # Extraer Explanation objects para el waterfall
        # explainer() retorna un objeto de tipo shap.Explanation.
        # check_additivity=False por el mismo motivo que en compute_shap_values:
        # el link 'count:poisson' rompe la comprobación de aditividad.
        exp_home = self.explainer_home(match_features, check_additivity=False)
        exp_away = self.explainer_away(match_features, check_additivity=False)

        # --- Waterfall Plot Local ---
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp_home[0], show=False)
        plt.title(f"Interpretación Local - {team_home} (vs {team_away})")
        plt.tight_layout()
        home_path = os.path.join(save_dir, f"waterfall_{team_home}_vs_{team_away}_home.png")
        # Sanear nombre de archivo
        home_path = home_path.replace(" ", "_")
        plt.savefig(home_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(os.path.abspath(home_path))

        # --- Waterfall Plot Visitante ---
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp_away[0], show=False)
        plt.title(f"Interpretación Visitante - {team_away} (vs {team_home})")
        plt.tight_layout()
        away_path = os.path.join(save_dir, f"waterfall_{team_home}_vs_{team_away}_away.png")
        away_path = away_path.replace(" ", "_")
        plt.savefig(away_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(os.path.abspath(away_path))

        logger.info(f"Waterfall plots generados para {team_home} vs {team_away}")
        return saved_files
