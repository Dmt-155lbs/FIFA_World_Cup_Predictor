"""
Módulo de modelos ML para el predictor del Mundial FIFA 2026.

Exporta los componentes principales del pipeline de modelado:
- XGBoostTrainer: Entrenamiento de modelos XGBoost para predicción de goles.
- OptunaOptimizer: Optimización de hiperparámetros con Optuna.
- ExperimentTracker: Seguimiento de experimentos con MLflow.
- BivariatePoisson: Modelo Poisson bivariado con corrección Dixon-Coles.
"""

from src.models.xgboost_trainer import XGBoostTrainer
from src.models.optuna_optimizer import OptunaOptimizer
from src.models.mlflow_tracking import ExperimentTracker
from src.models.poisson_model import BivariatePoisson

__all__ = [
    "XGBoostTrainer",
    "OptunaOptimizer",
    "ExperimentTracker",
    "BivariatePoisson",
]
