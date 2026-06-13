"""
Entrenador XGBoost para predicción de goles esperados (λ) en partidos de fútbol.

Entrena dos modelos independientes (local y visitante) usando el objetivo
count:poisson para modelar la distribución de goles como un proceso de Poisson.
La división de datos usa split temporal para evitar filtración de datos futuros.

Autor: Mundial 2026 Team
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from xgboost import XGBRegressor

from src.config import get_settings

log = structlog.get_logger(__name__)


class XGBoostTrainer:
    """
    Entrenador dual de XGBRegressor para predicción de goles locales y visitantes.

    Utiliza el objetivo 'count:poisson' que modela la variable respuesta como
    una distribución de Poisson, ideal para conteos de goles. Los modelos se
    entrenan con split temporal para respetar la causalidad de los datos.

    Atributos:
        params: Diccionario de hiperparámetros para XGBoost.
        model_home: Modelo entrenado para goles del equipo local.
        model_away: Modelo entrenado para goles del equipo visitante.
    """

    # ====================================================================== #
    #  CONSTRAINTS MONÓTONAS (candado lógico futbolístico)                     #
    # ====================================================================== #
    # Fuerzan que el modelo respete la dirección causal de cada feature de
    # fuerza, impidiendo que la "regresión a la media" de los árboles + la
    # dominancia de fifa_attack_diff inflen las probabilidades del underdog.
    #
    # Hay DOS modelos y las direcciones SE ESPEJAN entre ellos:
    #   - GOLES LOCALES  ↑ con fuerza/ataque LOCAL  y ↓ con fuerza/defensa VISITANTE.
    #   - GOLES VISITANTE ↑ con fuerza/ataque VISITANTE y ↓ con fuerza/defensa LOCAL.
    # Convención XGBoost: +1 = monótona creciente, -1 = decreciente, 0/omitido = sin
    # restricción. Las features de contexto/categóricas (confederación, descanso,
    # asistencia, competition_weight) quedan SIN restringir: no tienen una
    # dirección causal clara sobre los goles. Las de ataque sólo restringen el
    # marcador PROPIO; las de defensa, el marcador del RIVAL; overall/elo (calidad
    # global) restringen ambos lados (propio +, rival -).

    # Restricciones para el modelo de GOLES LOCALES.
    _MONOTONE_HOME: dict[str, int] = {
        # Calidad global (sube goles propios)
        "home_elo": +1, "away_elo": -1, "elo_diff": +1,
        "home_fifa_overall": +1, "away_fifa_overall": -1,
        "home_fifa_overall_copy": +1, "away_fifa_overall_copy": -1,
        # Ataque local → más goles locales; defensa visitante → menos goles locales
        "home_fifa_attack": +1, "away_fifa_defence": -1,
        "home_fifa_midfield": +1,
        # xG del partido / forma reciente (local marca, visitante encaja)
        "home_xg": +1, "away_xga": +1, "home_npxg": +1,
        "home_rolling_goals_scored": +1, "away_rolling_goals_conceded": +1,
        "home_rolling_xg": +1, "away_rolling_xga": +1,
        "home_rolling_form": +1, "home_rolling_elo_momentum": +1,
        # Diferenciales (home − away): a favor del local
        "xg_diff": +1, "form_diff": +1, "fifa_attack_diff": +1, "goals_diff": +1,
    }

    # Restricciones para el modelo de GOLES VISITANTES (espejo del anterior).
    _MONOTONE_AWAY: dict[str, int] = {
        "home_elo": -1, "away_elo": +1, "elo_diff": -1,
        "home_fifa_overall": -1, "away_fifa_overall": +1,
        "home_fifa_overall_copy": -1, "away_fifa_overall_copy": +1,
        "away_fifa_attack": +1, "home_fifa_defence": -1,
        "away_fifa_midfield": +1,
        "away_xg": +1, "home_xga": +1, "away_npxg": +1,
        "away_rolling_goals_scored": +1, "home_rolling_goals_conceded": +1,
        "away_rolling_xg": +1, "home_rolling_xga": +1,
        "away_rolling_form": +1, "away_rolling_elo_momentum": +1,
        "xg_diff": -1, "form_diff": -1, "fifa_attack_diff": -1, "goals_diff": -1,
    }

    @classmethod
    def monotone_constraints(
        cls, feature_names: list[str], side: str
    ) -> dict[str, int]:
        """Construye el dict de constraints para un modelo, filtrado a las
        columnas realmente presentes (XGBoost exige nombres existentes).

        Parameters
        ----------
        feature_names : list[str]
            Columnas del DataFrame de entrenamiento.
        side : str
            ``"home"`` o ``"away"``.
        """
        spec = cls._MONOTONE_HOME if side == "home" else cls._MONOTONE_AWAY
        cols = set(feature_names)
        return {f: c for f, c in spec.items() if f in cols}

    # Parámetros por defecto optimizados para predicción de goles
    _DEFAULT_PARAMS: dict[str, Any] = {
        "tree_method": "hist",           # Solo CPU
        "objective": "count:poisson",    # Distribución de Poisson para conteos
        "eval_metric": "poisson-nloglik",  # Log-verosimilitud negativa de Poisson
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Inicializa el entrenador con hiperparámetros opcionales.

        Args:
            params: Diccionario de hiperparámetros para XGBoost. Si es None,
                    se usan los parámetros por defecto. Si se proporciona, se
                    fusionan con los valores por defecto (los proporcionados
                    tienen prioridad).
        """
        settings = get_settings()
        self.params: dict[str, Any] = {**self._DEFAULT_PARAMS}
        # Asegurar que tree_method siempre viene de la configuración
        self.params["tree_method"] = settings.xgb_tree_method
        if params is not None:
            self.params.update(params)

        self.model_home: XGBRegressor | None = None
        self.model_away: XGBRegressor | None = None

        log.info(
            "Entrenador XGBoost inicializado",
            tree_method=self.params["tree_method"],
            objective=self.params["objective"],
            n_estimators=self.params["n_estimators"],
        )

    def train(
        self,
        X: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
        eval_fraction: float = 0.2,
    ) -> dict[str, float]:
        """
        Entrena dos modelos XGBoost: uno para goles locales y otro para visitantes.

        Usa split temporal: las últimas `eval_fraction` filas (ordenadas por
        fecha/índice) se reservan para evaluación con early stopping.

        Args:
            X: DataFrame de features. Debe estar ordenado cronológicamente.
            y_home: Serie con goles del equipo local.
            y_away: Serie con goles del equipo visitante.
            eval_fraction: Fracción de datos para el conjunto de evaluación
                          (por defecto 0.2 = 20% más reciente).

        Returns:
            Diccionario con métricas de entrenamiento y evaluación:
            - train_poisson_nloglik_home, eval_poisson_nloglik_home
            - train_poisson_nloglik_away, eval_poisson_nloglik_away
            - n_train, n_eval

        Raises:
            ValueError: Si X, y_home o y_away están vacíos o tienen tamaños
                       inconsistentes.
        """
        if len(X) == 0:
            raise ValueError("El DataFrame de features está vacío.")
        if len(X) != len(y_home) or len(X) != len(y_away):
            raise ValueError(
                f"Tamaños inconsistentes: X={len(X)}, "
                f"y_home={len(y_home)}, y_away={len(y_away)}"
            )

        # --- División temporal (sin aleatorización) ---
        n_total = len(X)
        n_eval = max(1, int(n_total * eval_fraction))
        n_train = n_total - n_eval

        X_train, X_eval = X.iloc[:n_train], X.iloc[n_train:]
        y_home_train, y_home_eval = y_home.iloc[:n_train], y_home.iloc[n_train:]
        y_away_train, y_away_eval = y_away.iloc[:n_train], y_away.iloc[n_train:]

        log.info(
            "División temporal realizada",
            n_train=n_train,
            n_eval=n_eval,
            eval_fraction=eval_fraction,
        )

        # --- Constraints monótonas por modelo (candado lógico) ---
        # Base sin monotone_constraints (se inyecta distinto por modelo).
        base_params = {
            k: v for k, v in self.params.items() if k != "monotone_constraints"
        }
        feat_names = list(X.columns)
        cons_home = self.monotone_constraints(feat_names, "home")
        cons_away = self.monotone_constraints(feat_names, "away")
        log.info(
            "Constraints monótonas aplicadas",
            n_home=len(cons_home),
            n_away=len(cons_away),
        )

        # --- Entrenamiento del modelo local ---
        self.model_home = XGBRegressor(
            **base_params,
            monotone_constraints=cons_home,
            early_stopping_rounds=50,
        )
        self.model_home.fit(
            X_train,
            y_home_train,
            eval_set=[(X_eval, y_home_eval)],
            verbose=False,
        )

        # --- Entrenamiento del modelo visitante ---
        self.model_away = XGBRegressor(
            **base_params,
            monotone_constraints=cons_away,
            early_stopping_rounds=50,
        )
        self.model_away.fit(
            X_train,
            y_away_train,
            eval_set=[(X_eval, y_away_eval)],
            verbose=False,
        )

        # --- Recopilación de métricas ---
        metrics: dict[str, float] = {
            "n_train": float(n_train),
            "n_eval": float(n_eval),
        }

        # Métricas del modelo local
        home_results = self.model_home.evals_result()
        home_metric_key = list(home_results["validation_0"].keys())[0]
        metrics["eval_poisson_nloglik_home"] = home_results["validation_0"][
            home_metric_key
        ][-1]
        metrics["train_poisson_nloglik_home"] = float(
            self.model_home.best_score
        )

        # Métricas del modelo visitante
        away_results = self.model_away.evals_result()
        away_metric_key = list(away_results["validation_0"].keys())[0]
        metrics["eval_poisson_nloglik_away"] = away_results["validation_0"][
            away_metric_key
        ][-1]
        metrics["train_poisson_nloglik_away"] = float(
            self.model_away.best_score
        )

        log.info(
            "Entrenamiento completado",
            eval_nloglik_home=metrics["eval_poisson_nloglik_home"],
            eval_nloglik_away=metrics["eval_poisson_nloglik_away"],
        )

        return metrics

    def predict(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predice los goles esperados (λ) para equipos locales y visitantes.

        Los valores se recortan a un mínimo de 0.01 para evitar lambdas
        no positivos que causarían problemas en la distribución de Poisson.

        Args:
            X: DataFrame de features para predicción.

        Returns:
            Tupla (lambda_home, lambda_away) con arrays de predicciones
            positivas (≥ 0.01).

        Raises:
            RuntimeError: Si los modelos no han sido entrenados.
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError(
                "Los modelos no han sido entrenados. Ejecute train() primero."
            )

        lambda_home: np.ndarray = self.model_home.predict(X)
        lambda_away: np.ndarray = self.model_away.predict(X)

        # Recortar a mínimo 0.01 para estabilidad numérica
        lambda_home = np.clip(lambda_home, a_min=0.01, a_max=None)
        lambda_away = np.clip(lambda_away, a_min=0.01, a_max=None)

        log.debug(
            "Predicción de lambdas realizada",
            n_muestras=len(X),
            lambda_home_media=float(np.mean(lambda_home)),
            lambda_away_media=float(np.mean(lambda_away)),
        )

        return lambda_home, lambda_away

    def save(self, path: str) -> None:
        """
        Guarda ambos modelos (local y visitante) en un único archivo joblib.

        Args:
            path: Ruta del archivo de destino (.joblib).

        Raises:
            RuntimeError: Si los modelos no han sido entrenados.
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError(
                "No se pueden guardar modelos no entrenados."
            )

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_home": self.model_home,
            "model_away": self.model_away,
            "params": self.params,
        }
        joblib.dump(payload, save_path)

        log.info("Modelos guardados exitosamente", ruta=str(save_path))

    def load(self, path: str) -> None:
        """
        Carga ambos modelos desde un archivo joblib.

        Args:
            path: Ruta del archivo fuente (.joblib).

        Raises:
            FileNotFoundError: Si el archivo no existe.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de modelos: {load_path}"
            )

        payload: dict[str, Any] = joblib.load(load_path)
        self.model_home = payload["model_home"]
        self.model_away = payload["model_away"]
        self.params = payload.get("params", self.params)

        log.info("Modelos cargados exitosamente", ruta=str(load_path))

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calcula la importancia de features combinada de ambos modelos.

        Promedia las importancias 'gain' de los modelos local y visitante
        para obtener una vista unificada de qué features son más relevantes.

        Returns:
            DataFrame con columnas ['feature', 'importance_home',
            'importance_away', 'importance_mean'] ordenado por
            importance_mean descendente.

        Raises:
            RuntimeError: Si los modelos no han sido entrenados.
        """
        if self.model_home is None or self.model_away is None:
            raise RuntimeError(
                "Los modelos no han sido entrenados. "
                "Ejecute train() primero."
            )

        # Obtener importancias tipo 'gain' de ambos modelos
        imp_home = self.model_home.feature_importances_
        imp_away = self.model_away.feature_importances_

        feature_names = self.model_home.get_booster().feature_names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imp_home))]

        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance_home": imp_home,
            "importance_away": imp_away,
        })
        df_importance["importance_mean"] = (
            df_importance["importance_home"] + df_importance["importance_away"]
        ) / 2.0

        df_importance = df_importance.sort_values(
            "importance_mean", ascending=False
        ).reset_index(drop=True)

        log.info(
            "Importancia de features calculada",
            n_features=len(df_importance),
            top_feature=df_importance.iloc[0]["feature"]
            if len(df_importance) > 0
            else "N/A",
        )

        return df_importance
