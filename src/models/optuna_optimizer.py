"""
Optimizador de hiperparámetros para XGBoost usando Optuna.

Implementa búsqueda bayesiana de hiperparámetros con validación cruzada
temporal (TimeSeriesSplit) para respetar la causalidad de los datos deportivos.
La métrica de optimización es la desviación de Poisson media entre los folds
para ambos modelos (local y visitante).

Autor: Mundial 2026 Team
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import structlog
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.config import get_settings

log = structlog.get_logger(__name__)


class OptunaOptimizer:
    """
    Optimizador bayesiano de hiperparámetros para el entrenador XGBoost.

    Utiliza Optuna con validación cruzada temporal (TimeSeriesSplit con 5
    splits) para encontrar los hiperparámetros óptimos que minimicen la
    desviación de Poisson media combinada (local + visitante).

    Atributos:
        n_trials: Número máximo de trials para la búsqueda.
        timeout: Tiempo máximo en segundos para la optimización.
    """

    def __init__(
        self,
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Inicializa el optimizador con configuración de búsqueda.

        Args:
            n_trials: Número máximo de trials. Si es None, se usa el valor
                     de la configuración (optuna_n_trials, default=100).
            timeout: Tiempo máximo en segundos. Si es None, se usa el valor
                    de la configuración (optuna_timeout_seconds, default=3600).
        """
        settings = get_settings()
        self.n_trials: int = (
            n_trials
            if n_trials is not None
            else getattr(settings, "optuna_n_trials", 100)
        )
        self.timeout: int = (
            timeout
            if timeout is not None
            else getattr(settings, "optuna_timeout_seconds", 3600)
        )

        log.info(
            "Optimizador Optuna inicializado",
            n_trials=self.n_trials,
            timeout_seconds=self.timeout,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
    ) -> float:
        """
        Función objetivo para un trial individual de Optuna.

        Sugiere hiperparámetros, entrena con validación cruzada temporal
        (5 folds) y retorna la desviación de Poisson media combinada
        de ambos modelos.

        Args:
            trial: Objeto Trial de Optuna con los hiperparámetros sugeridos.
            X: DataFrame de features (ordenado cronológicamente).
            y_home: Serie con goles del equipo local.
            y_away: Serie con goles del equipo visitante.

        Returns:
            Desviación de Poisson media combinada (local + visitante)
            promediada sobre todos los folds de validación cruzada.
        """
        # --- Sugerencia de hiperparámetros ---
        params: dict[str, Any] = {
            "tree_method": "hist",
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "verbosity": 0,
            "random_state": 42,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }

        # --- Validación cruzada temporal ---
        tscv = TimeSeriesSplit(n_splits=5)
        deviances: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_home_train = y_home.iloc[train_idx]
            y_home_val = y_home.iloc[val_idx]
            y_away_train = y_away.iloc[train_idx]
            y_away_val = y_away.iloc[val_idx]

            # Modelo local
            model_home = XGBRegressor(**params)
            model_home.fit(
                X_train,
                y_home_train,
                eval_set=[(X_val, y_home_val)],
                verbose=False,
            )

            # Modelo visitante
            model_away = XGBRegressor(**params)
            model_away.fit(
                X_train,
                y_away_train,
                eval_set=[(X_val, y_away_val)],
                verbose=False,
            )

            # Predicciones recortadas para estabilidad
            pred_home = np.clip(model_home.predict(X_val), 0.01, None)
            pred_away = np.clip(model_away.predict(X_val), 0.01, None)

            # Desviación de Poisson por fold (media de local + visitante)
            dev_home = mean_poisson_deviance(y_home_val, pred_home)
            dev_away = mean_poisson_deviance(y_away_val, pred_away)
            fold_deviance = (dev_home + dev_away) / 2.0
            deviances.append(fold_deviance)

            # Poda intermedia: reportar resultado parcial
            trial.report(np.mean(deviances), fold_idx)
            if trial.should_prune():
                log.debug(
                    "Trial podado por rendimiento insuficiente",
                    trial_number=trial.number,
                    fold=fold_idx,
                )
                raise optuna.TrialPruned()

        mean_deviance = float(np.mean(deviances))

        log.debug(
            "Trial completado",
            trial_number=trial.number,
            mean_deviance=mean_deviance,
        )

        return mean_deviance

    def optimize(
        self,
        X: pd.DataFrame,
        y_home: pd.Series,
        y_away: pd.Series,
    ) -> dict[str, Any]:
        """
        Ejecuta la optimización bayesiana de hiperparámetros.

        Crea un estudio Optuna con dirección 'minimize' (minimizar la
        desviación de Poisson) y ejecuta los trials configurados.

        Args:
            X: DataFrame de features (ordenado cronológicamente).
            y_home: Serie con goles del equipo local.
            y_away: Serie con goles del equipo visitante.

        Returns:
            Diccionario con los mejores hiperparámetros encontrados,
            incluyendo las claves fijas (tree_method, objective, etc.)
            listas para pasar directamente a XGBoostTrainer.
        """
        log.info(
            "Iniciando optimización de hiperparámetros",
            n_trials=self.n_trials,
            timeout_seconds=self.timeout,
            n_muestras=len(X),
            n_features=X.shape[1],
        )

        # Crear estudio con poda mediana
        study = optuna.create_study(
            direction="minimize",
            study_name="xgboost-poisson-optimization",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=2,
            ),
        )

        # Suprimir logs verbosos de Optuna durante la optimización
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            lambda trial: self._objective(trial, X, y_home, y_away),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        # Construir diccionario de mejores parámetros con claves fijas
        best_params: dict[str, Any] = {
            "tree_method": "hist",
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "verbosity": 0,
            "random_state": 42,
            **study.best_params,
        }

        log.info(
            "Optimización completada",
            mejores_params=best_params,
            mejor_deviance=study.best_value,
            trials_completados=len(study.trials),
            trials_podados=len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]),
        )

        return best_params
