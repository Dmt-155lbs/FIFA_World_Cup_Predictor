"""
Pipeline orquestador del Ensemble Híbrido de 3 capas.

Encadena secuencialmente:
    1. Feature Engineering → preparación de datos
    2. Capa Macro: XGBoost → estimación de lambdas (fuerza ofensiva/defensiva)
    3. Capa Estocástica: Poisson Bivariado → matriz de probabilidades de marcadores
    4. Capa Estructural: Monte Carlo → simulación del torneo completo

Soporta dos modos de ejecución:
    - Entrenamiento: optimiza hiperparámetros (Optuna) y entrena XGBoost desde cero.
    - Inferencia: carga el mejor modelo y genera predicciones/simulaciones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import os
import shutil
import tempfile

from src.config import get_settings
from src.features.feature_engineering import FeatureEngineer
from src.explainability.shap_analysis import SHAPAnalyzer
from src.models.xgboost_trainer import XGBoostTrainer
from src.models.optuna_optimizer import OptunaOptimizer
from src.models.mlflow_tracking import ExperimentTracker
from src.models.poisson_model import BivariatePoisson
from src.simulation.bracket_engine import BracketEngine
from src.simulation.monte_carlo import MonteCarloSimulator, SimulationResults
from src.utils.constants import TARGET_HOME, TARGET_AWAY

logger = structlog.get_logger(__name__)


@dataclass
class MatchPrediction:
    """Predicción completa para un partido individual."""
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    prob_home: float
    prob_draw: float
    prob_away: float
    prob_over_25: float
    prob_under_25: float
    most_likely_scores: list[tuple[int, int, float]]
    expected_home_goals: float
    expected_away_goals: float


class PredictionPipeline:
    """
    Pipeline principal que orquesta las 3 capas del Ensemble Híbrido.

    Uso típico (entrenamiento):
        pipeline = PredictionPipeline()
        pipeline.train()

    Uso típico (inferencia de un partido):
        pipeline = PredictionPipeline(model_path='models/best_model')
        pred = pipeline.predict_match(features_partido)

    Uso típico (simulación de torneo):
        pipeline = PredictionPipeline(model_path='models/best_model')
        results = pipeline.simulate_tournament(team_lambdas)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        engine=None,
    ):
        """
        Inicializa el pipeline con sus componentes.

        Args:
            model_path: Ruta al modelo entrenado. Si se proporciona, se carga
                        para modo inferencia. Si es None, se asume modo entrenamiento.
            engine: Motor SQLAlchemy opcional. Si es None, usa el engine global.
        """
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineer(engine=engine)
        self.trainer = XGBoostTrainer()
        self.poisson = BivariatePoisson(
            max_goals=self.settings.max_goals_matrix,
            rho=self.settings.poisson_rho_default,
        )
        self.bracket_engine = BracketEngine()
        self.tracker = ExperimentTracker()

        # Cargar modelo existente si se provee ruta
        if model_path and Path(model_path).exists():
            self.trainer.load(model_path)
            logger.info(
                "Modelo cargado desde disco para inferencia.",
                ruta=model_path,
            )

    # =========================================================================
    # Fase 1: Preparación de Features
    # =========================================================================

    def prepare_features(
        self, df: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Ejecuta el pipeline completo de feature engineering.

        Args:
            df: DataFrame opcional con datos crudos. Si es None, carga desde
                la vista vw_feature_store de la base de datos.

        Returns:
            Tupla (X_features, y_home_goals, y_away_goals) lista para XGBoost.
        """
        logger.info("Iniciando preparación de features...")

        if df is None:
            df = self.feature_engineer.load_feature_store()

        X, y_home, y_away = self.feature_engineer.build_features(df)

        logger.info(
            "Features preparados exitosamente.",
            filas=len(X),
            columnas=len(X.columns),
        )
        return X, y_home, y_away

    # =========================================================================
    # Fase 2: Entrenamiento del Modelo
    # =========================================================================

    def train(
        self,
        X: Optional[pd.DataFrame] = None,
        y_home: Optional[pd.Series] = None,
        y_away: Optional[pd.Series] = None,
        optimize_hyperparams: bool = True,
        save_path: str = './models/best_model',
        persist: bool = True,
    ) -> dict:
        """
        Entrena el modelo XGBoost completo con optimización opcional de
        hiperparámetros vía Optuna y tracking en MLflow.

        Args:
            X: Features. Si es None, ejecuta prepare_features() primero.
            y_home: Target de goles del equipo local.
            y_away: Target de goles del equipo visitante.
            optimize_hyperparams: Si True, ejecuta Optuna antes de entrenar.
            save_path: Ruta donde guardar el modelo entrenado.

        Returns:
            Dict con métricas de evaluación del modelo.
        """
        logger.info("Iniciando pipeline de entrenamiento...")

        # Preparar features si no se proporcionan
        if X is None or y_home is None or y_away is None:
            X, y_home, y_away = self.prepare_features()

        # Optimización de hiperparámetros con Optuna
        best_params = {}
        if optimize_hyperparams:
            logger.info(
                "Ejecutando optimización de hiperparámetros con Optuna.",
                n_trials=self.settings.optuna_n_trials,
                timeout=self.settings.optuna_timeout_seconds,
            )
            optimizer = OptunaOptimizer(
                n_trials=self.settings.optuna_n_trials,
                timeout=self.settings.optuna_timeout_seconds,
            )
            best_params = optimizer.optimize(X, y_home, y_away)
            logger.info(
                "Optimización completada.",
                mejores_params=best_params,
            )

        # Configurar trainer con los mejores parámetros
        if best_params:
            self.trainer = XGBoostTrainer(params=best_params)

        # Entrenar el modelo (siempre)
        metrics = self.trainer.train(X, y_home, y_away)
        logger.info("Modelo entrenado.", métricas=metrics)

        # Persistencia + tracking. Se OMITE durante la evaluación walk-forward
        # (persist=False): allí se entrena un modelo por fold solo para medir,
        # y no queremos (a) sobrescribir el modelo de producción en disco,
        # (b) registrar una versión por fold en la BD, ni (c) abrir un run +
        # SHAP por fold.
        if persist:
            # Iniciar tracking en MLflow
            run_id = self.tracker.start_run(
                run_name="entrenamiento_ensemble",
                params=best_params if best_params else self.trainer.params,
            )

            # Logear métricas y modelo en MLflow
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(self.trainer, "xgboost_ensemble")

            # --- Análisis SHAP y registro de artefactos ---
            # El summary y el waterfall se aíslan en bloques try/except
            # independientes: si uno falla, el otro igual se registra. Se
            # garantiza además un artefacto con nombre canónico
            # (shap_summary.png / shap_waterfall.png) que el dashboard busca.
            try:
                logger.info("Iniciando análisis SHAP...")
                feature_names = list(X.columns)
                shap_analyzer = SHAPAnalyzer(
                    model_home=self.trainer.model_home,
                    model_away=self.trainer.model_away,
                    feature_names=feature_names,
                )

                # SHAP sobre TreeExplainer es exacto pero el render del summary
                # con miles de puntos es costoso; se acota la muestra para
                # mantener el entrenamiento ágil sin perder representatividad.
                X_shap = (
                    X.sample(n=2000, random_state=42) if len(X) > 2000 else X
                )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # --- Summary plots (importancia global) ---
                    try:
                        summary_files = shap_analyzer.plot_summary(
                            X_shap, save_dir=tmp_dir
                        )
                        # Renombrar el plot del modelo local al nombre canónico
                        # que consume el dashboard (shap_summary.png) y conservar
                        # el del visitante como variante.
                        if summary_files:
                            canonical = os.path.join(tmp_dir, "shap_summary.png")
                            shutil.move(summary_files[0], canonical)
                            summary_files = [canonical] + summary_files[1:]
                        for f in summary_files:
                            self.tracker.log_artifact(f, "shap_summary")
                        logger.info(
                            "SHAP summary registrado.", archivos=len(summary_files)
                        )
                    except Exception as e:
                        logger.error("Error en SHAP summary", error=str(e))

                    # --- Waterfall plot para un partido de muestra ---
                    try:
                        if len(X) > 0:
                            sample_features = X.iloc[[0]]
                            waterfall_files = shap_analyzer.explain_match(
                                match_features=sample_features,
                                team_home="Local_Sample",
                                team_away="Visitante_Sample",
                                save_dir=tmp_dir,
                            )
                            if waterfall_files:
                                canonical = os.path.join(
                                    tmp_dir, "shap_waterfall.png"
                                )
                                shutil.move(waterfall_files[0], canonical)
                                waterfall_files = (
                                    [canonical] + waterfall_files[1:]
                                )
                            for f in waterfall_files:
                                self.tracker.log_artifact(f, "shap_waterfall")
                            logger.info(
                                "SHAP waterfall registrado.",
                                archivos=len(waterfall_files),
                            )
                    except Exception as e:
                        logger.error("Error en SHAP waterfall", error=str(e))

                logger.info("Análisis SHAP completado.")
            except Exception as e:
                logger.error("Error durante el análisis SHAP", error=str(e))

            self.tracker.end_run()

            # Guardar modelo en disco
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.trainer.save(save_path)
            logger.info("Modelo guardado.", ruta=save_path)

            # Registrar en la base de datos
            self.tracker.register_to_db(
                run_id=run_id,
                version_tag=f"v{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                metrics=metrics,
                artifact_path=save_path,
            )

        # Calibrar rho de Dixon-Coles con los datos de entrenamiento (siempre:
        # necesario para que las predicciones de cada fold sean válidas).
        self._calibrate_poisson(X, y_home, y_away)

        return metrics

    def _calibrate_poisson(
        self,
        X: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
    ) -> None:
        """
        Calibra el parámetro rho de Dixon-Coles usando los lambdas predichos
        por XGBoost y los resultados reales.
        """
        logger.info("Calibrando parámetro rho de Dixon-Coles...")

        lambda_home, lambda_away = self.trainer.predict(X)
        historical_lambdas = np.column_stack([lambda_home, lambda_away])
        historical_results = np.column_stack([y_home.values, y_away.values])

        optimal_rho = self.poisson.fit_rho(historical_lambdas, historical_results)
        self.poisson.rho = optimal_rho

        logger.info("Rho calibrado.", rho_optimo=optimal_rho)

    # =========================================================================
    # Fase 3: Predicción de Partido Individual
    # =========================================================================

    def predict_match(
        self,
        match_features: pd.DataFrame,
        home_team: str = "Local",
        away_team: str = "Visitante",
    ) -> MatchPrediction:
        """
        Genera la predicción completa para un partido individual pasando
        por las 3 capas del Ensemble.

        Args:
            match_features: DataFrame con una fila de features del partido.
            home_team: Nombre del equipo local (para el reporte).
            away_team: Nombre del equipo visitante (para el reporte).

        Returns:
            MatchPrediction con todas las probabilidades y marcadores.
        """
        # Capa 1: XGBoost → lambdas
        lambda_home_arr, lambda_away_arr = self.trainer.predict(match_features)
        lambda_home = float(lambda_home_arr[0])
        lambda_away = float(lambda_away_arr[0])

        # Capa 2: Poisson Bivariado → matriz de probabilidades
        matrix = self.poisson.score_matrix(lambda_home, lambda_away)
        probs = self.poisson.match_probabilities(matrix)
        ou_probs = self.poisson.over_under_probs(matrix)
        exp_goals = self.poisson.expected_goals(matrix)
        top_scores = self.poisson.most_likely_scores(matrix)

        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            prob_home=probs['prob_home'],
            prob_draw=probs['prob_draw'],
            prob_away=probs['prob_away'],
            prob_over_25=ou_probs['prob_over'],
            prob_under_25=ou_probs['prob_under'],
            most_likely_scores=top_scores,
            expected_home_goals=exp_goals[0],
            expected_away_goals=exp_goals[1],
        )

        logger.info(
            "Predicción generada.",
            partido=f"{home_team} vs {away_team}",
            prob_local=f"{probs['prob_home']:.1%}",
            prob_empate=f"{probs['prob_draw']:.1%}",
            prob_visitante=f"{probs['prob_away']:.1%}",
        )

        return prediction

    # =========================================================================
    # Fase 4: Simulación del Torneo Completo
    # =========================================================================

    def simulate_tournament(
        self,
        team_lambdas: dict[str, tuple[float, float]],
        n_simulations: Optional[int] = None,
    ) -> SimulationResults:
        """
        Ejecuta la simulación Monte Carlo del torneo completo.

        Args:
            team_lambdas: Dict {nombre_equipo: (lambda_ofensivo, lambda_defensivo)}
                          para los 48 equipos del Mundial.
            n_simulations: Número de simulaciones. Si es None, usa la
                           configuración por defecto.

        Returns:
            SimulationResults con probabilidades de campeón, avance por ronda, etc.
        """
        n_sims = n_simulations or self.settings.n_simulations

        logger.info(
            "Iniciando simulación Monte Carlo del torneo.",
            simulaciones=n_sims,
            equipos=len(team_lambdas),
        )

        simulator = MonteCarloSimulator(
            bracket_engine=self.bracket_engine,
            seed=self.settings.mc_random_seed,
        )

        results = simulator.simulate(
            team_lambdas=team_lambdas,
            n_simulations=n_sims,
        )

        logger.info(
            "Simulación completada.",
            equipos_con_probabilidades=len(results.champion_probs),
            campeon_mas_probable=max(
                results.champion_probs, key=results.champion_probs.get
            ) if results.champion_probs else "N/A",
        )

        return results

    # =========================================================================
    # Pipeline Completo de Extremo a Extremo
    # =========================================================================

    def run_full_pipeline(
        self,
        team_lambdas: Optional[dict[str, tuple[float, float]]] = None,
        optimize: bool = True,
        save_path: str = './models/best_model',
        n_simulations: Optional[int] = None,
    ) -> tuple[dict, Optional[SimulationResults]]:
        """
        Ejecuta el pipeline completo: entrenamiento + simulación.

        Args:
            team_lambdas: Lambdas por equipo para simulación. Si es None,
                          solo entrena sin simular.
            optimize: Si True, ejecuta optimización Optuna.
            save_path: Ruta para guardar el modelo.
            n_simulations: Número de simulaciones Monte Carlo.

        Returns:
            Tupla (métricas_entrenamiento, resultados_simulación).
        """
        logger.info("="*60)
        logger.info("INICIANDO PIPELINE COMPLETO DEL ENSEMBLE HÍBRIDO")
        logger.info("="*60)

        # Entrenar
        metrics = self.train(optimize_hyperparams=optimize, save_path=save_path)

        # Simular torneo si se proveen lambdas
        sim_results = None
        if team_lambdas:
            sim_results = self.simulate_tournament(
                team_lambdas=team_lambdas,
                n_simulations=n_simulations,
            )

        logger.info("Pipeline completo finalizado exitosamente.")
        return metrics, sim_results
