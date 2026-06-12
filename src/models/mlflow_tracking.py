"""
Seguimiento de experimentos ML con MLflow y registro en base de datos.

Proporciona una interfaz unificada para:
- Registrar parámetros, métricas y artefactos de modelos en MLflow.
- Persistir metadatos de versiones de modelos en las tablas
  ML_EXPERIMENT y ML_MODEL_VERSION de la base de datos SQL.

Autor: Mundial 2026 Team
"""

from __future__ import annotations


from typing import Any

import mlflow
import structlog
from sqlalchemy import text

from src.config import get_settings
from src.utils.db import get_session

log = structlog.get_logger(__name__)


class ExperimentTracker:
    """
    Gestor de seguimiento de experimentos con MLflow y persistencia en BD.

    Encapsula las operaciones de MLflow (inicio/fin de runs, logging de
    params/métricas/modelos) y proporciona un método para registrar
    resultados en las tablas SQL del proyecto.

    Atributos:
        experiment_name: Nombre del experimento en MLflow.
        _active_run_id: ID del run activo (None si no hay run activo).
    """

    def __init__(
        self, experiment_name: str = "mundial-2026-predictor"
    ) -> None:
        """
        Inicializa el tracker configurando la URI de MLflow y el experimento.

        Args:
            experiment_name: Nombre del experimento en MLflow. Si no existe,
                            se crea automáticamente.
        """
        settings = get_settings()
        self.experiment_name: str = experiment_name
        self._active_run_id: str | None = None

        # Configurar URI de tracking desde la configuración centralizada
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        log.info(
            "ExperimentTracker inicializado",
            experimento=self.experiment_name,
            tracking_uri=settings.mlflow_tracking_uri,
        )

    def start_run(
        self,
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """
        Inicia un nuevo run de MLflow y opcionalmente registra parámetros.

        Args:
            run_name: Nombre descriptivo del run (ej: 'train-v2.1-optuna').
            params: Diccionario de hiperparámetros a registrar. Si es None,
                   no se registran parámetros.

        Returns:
            ID único del run iniciado (run_id).
        """
        # Si ya hay un run activo (p. ej. la evaluación walk-forward abre un run
        # "paraguas" y luego entrena un modelo por fold, donde cada train inicia
        # su propio run), iniciamos uno ANIDADO en lugar de fallar con
        # "Run already active".
        nested = mlflow.active_run() is not None
        active_run = mlflow.start_run(run_name=run_name, nested=nested)
        self._active_run_id = active_run.info.run_id

        if params is not None:
            # MLflow requiere valores simples; convertir tipos complejos
            safe_params = {
                k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                for k, v in params.items()
            }
            mlflow.log_params(safe_params)

        log.info(
            "Run de MLflow iniciado",
            run_id=self._active_run_id,
            run_name=run_name,
            n_params=len(params) if params else 0,
        )

        return self._active_run_id

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Registra métricas en el run activo de MLflow.

        Args:
            metrics: Diccionario de métricas numéricas (ej:
                    {'poisson_nloglik_home': 0.85, 'mae_away': 1.2}).

        Raises:
            RuntimeError: Si no hay un run activo.
        """
        if self._active_run_id is None:
            raise RuntimeError(
                "No hay un run activo. Ejecute start_run() primero."
            )

        mlflow.log_metrics(metrics)

        log.info(
            "Métricas registradas en MLflow",
            run_id=self._active_run_id,
            n_metricas=len(metrics),
        )

    def log_model(self, model: Any, model_name: str) -> None:
        """
        Registra un artefacto de modelo en el run activo de MLflow.

        Usa mlflow.sklearn.log_model para serializar el modelo, lo cual
        es compatible con XGBRegressor gracias a la API de sklearn.

        Args:
            model: Objeto del modelo a registrar (XGBRegressor, dict, etc.).
            model_name: Nombre del artefacto en MLflow (ej: 'model_home').

        Raises:
            RuntimeError: Si no hay un run activo.
        """
        if self._active_run_id is None:
            raise RuntimeError(
                "No hay un run activo. Ejecute start_run() primero."
            )

        mlflow.sklearn.log_model(model, artifact_path=model_name)

        log.info(
            "Modelo registrado en MLflow",
            run_id=self._active_run_id,
            modelo=model_name,
        )

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Registra un archivo local como un artefacto en el run activo.

        Args:
            local_path: Ruta al archivo en el disco local.
            artifact_path: Directorio de destino dentro del run en MLflow.
                           Si es None, se guarda en la raíz de artefactos del run.

        Raises:
            RuntimeError: Si no hay un run activo.
        """
        if self._active_run_id is None:
            raise RuntimeError(
                "No hay un run activo. Ejecute start_run() primero."
            )

        mlflow.log_artifact(local_path, artifact_path)

        log.info(
            "Artefacto registrado en MLflow",
            run_id=self._active_run_id,
            archivo=local_path,
            destino=artifact_path,
        )

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """
        Registra todos los archivos de un directorio como artefactos en el run activo.

        Args:
            local_dir: Ruta al directorio en el disco local.
            artifact_path: Directorio de destino dentro del run en MLflow.

        Raises:
            RuntimeError: Si no hay un run activo.
        """
        if self._active_run_id is None:
            raise RuntimeError(
                "No hay un run activo. Ejecute start_run() primero."
            )

        mlflow.log_artifacts(local_dir, artifact_path)

        log.info(
            "Artefactos del directorio registrados en MLflow",
            run_id=self._active_run_id,
            directorio=local_dir,
            destino=artifact_path,
        )

    def end_run(self) -> None:
        """
        Finaliza el run activo de MLflow.

        Después de llamar a este método, se requiere un nuevo start_run()
        para registrar más datos.
        """
        if self._active_run_id is not None:
            mlflow.end_run()
            log.info(
                "Run de MLflow finalizado",
                run_id=self._active_run_id,
            )
            self._active_run_id = None
        else:
            log.warning("Se intentó finalizar un run sin run activo.")

    def register_to_db(
        self,
        run_id: str,
        version_tag: str,
        metrics: dict[str, float],
        artifact_path: str,
    ) -> None:
        """
        Registra los resultados del experimento en la base de datos SQL.

        Inserta registros en las tablas ML_EXPERIMENT y ML_MODEL_VERSION
        para mantener un historial persistente de versiones del modelo
        independiente de MLflow.

        Args:
            run_id: ID del run de MLflow asociado.
            version_tag: Etiqueta de versión del modelo (ej: 'v2.1.0').
            metrics: Diccionario de métricas del entrenamiento.
            artifact_path: Ruta al artefacto del modelo guardado.

        Raises:
            Exception: Si la inserción en BD falla (se hace rollback
                      automáticamente gracias al context manager).
        """
        with get_session() as session:
            # Insertar en ML_EXPERIMENT y recuperar el ID generado
            result = session.execute(
                text(
                    """
                    INSERT INTO [mundial].[ML_EXPERIMENT]
                    ([experiment_name], [mlflow_run_id])
                    OUTPUT INSERTED.[experiment_id]
                    VALUES (:experiment_name, :mlflow_run_id)
                    """
                ),
                {
                    "experiment_name": self.experiment_name,
                    "mlflow_run_id": run_id,
                },
            )
            experiment_id = result.scalar()

            # Insertar en ML_MODEL_VERSION vinculado al experimento
            session.execute(
                text(
                    """
                    INSERT INTO [mundial].[ML_MODEL_VERSION]
                    ([experiment_id], [version_tag], [brier_score],
                     [roi_backtest], [log_loss], [artifact_path])
                    VALUES (:experiment_id, :version_tag, :brier_score,
                            :roi_backtest, :log_loss, :artifact_path)
                    """
                ),
                {
                    "experiment_id": experiment_id,
                    "version_tag": version_tag,
                    "brier_score": metrics.get("brier_score", 0.0),
                    "roi_backtest": metrics.get("roi_backtest", 0.0),
                    "log_loss": metrics.get("log_loss", 0.0),
                    "artifact_path": artifact_path,
                },
            )

        log.info(
            "Resultados registrados en la base de datos",
            run_id=run_id,
            version_tag=version_tag,
            experiment_id=experiment_id,
            artifact_path=artifact_path,
        )
