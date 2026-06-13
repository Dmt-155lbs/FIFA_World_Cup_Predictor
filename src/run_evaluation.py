import os

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import KFold
from sqlalchemy import text
from src.pipeline import PredictionPipeline
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.financial_backtest import FinancialBacktester
from src.evaluation.metrics import generate_naive_baseline, compute_brier_score
from src.models.mlflow_tracking import ExperimentTracker
from src.models.probability_calibrator import (
    ProbabilityCalibrator,
    brier_multiclass,
)

logger = structlog.get_logger(__name__)

# Método y ruta del calibrador de probabilidades 1X2. Se persiste junto al
# modelo (mismo volumen models_data, que streamlit monta en solo-lectura) para
# que el dashboard de Value Bets pueda cargarlo y calcular el EV calibrado.
CALIBRATION_METHOD = "sigmoid"


def _calibrator_path() -> str:
    model_path = os.getenv("MODEL_PATH", "./models/best_model")
    return os.path.join(os.path.dirname(model_path) or ".", "calibrator.pkl")


def load_odds_map(engine) -> dict[int, dict[str, float]]:
    """Carga las cuotas 1X2 promedio por partido desde FACT_ODDS.

    Promedia las cuotas de todas las casas de apuestas registradas para cada
    partido (descartando cuotas centinela <= 0).

    Retorna
    -------
    dict[int, dict[str, float]]
        ``{match_id: {'home': cuota, 'draw': cuota, 'away': cuota}}``.
    """
    query = text(
        """
        SELECT [match_id],
               AVG([odds_home]) AS home,
               AVG([odds_draw]) AS draw,
               AVG([odds_away]) AS away
        FROM [mundial].[FACT_ODDS]
        WHERE [odds_home] > 0 AND [odds_draw] > 0 AND [odds_away] > 0
        GROUP BY [match_id]
        """
    )
    odds_map: dict[int, dict[str, float]] = {}
    try:
        with engine.connect() as conn:
            for row in conn.execute(query):
                odds_map[int(row.match_id)] = {
                    "home": float(row.home),
                    "draw": float(row.draw),
                    "away": float(row.away),
                }
    except Exception as exc:
        logger.warning("no_se_pudieron_cargar_cuotas", error=str(exc))
        return {}

    logger.info("Cuotas cargadas para backtesting", partidos_con_cuotas=len(odds_map))
    return odds_map


def run_offline_evaluation(
    splits: list[dict],
    full_data: pd.DataFrame,
    feature_cols: list[str],
    engine=None,
):
    """
    Script dedicado para ejecutar la evaluación Walk-Forward y el Backtesting.
    Esto se mantiene separado de pipeline.py porque el Walk-Forward reentrena el modelo
    múltiples veces (uno por fold) y es intensivo en cómputo.
    """
    logger.info("Iniciando Evaluación Offline Completa")
    
    pipeline = PredictionPipeline()
    evaluator = WalkForwardEvaluator(splits)
    tracker = ExperimentTracker()
    
    # Iniciar tracking de la evaluación global
    tracker.start_run(run_name="offline_walk_forward_evaluation")
    
    # 1. Ejecutar Walk-Forward
    logger.info("Ejecutando Walk-Forward...")
    wf_results = evaluator.evaluate(pipeline, full_data, feature_cols)
    
    # 2. Comparación contra el baseline Naive
    logger.info("Calculando Baseline Naive...")
    n_samples = len(wf_results['actuals'])
    naive_probs = generate_naive_baseline(n_samples)
    
    y_true_dummies = pd.get_dummies(wf_results['actuals'])
    # Asegurar orden
    for col in ['home', 'draw', 'away']:
        if col not in y_true_dummies.columns:
            y_true_dummies[col] = 0
    y_true_dummies = y_true_dummies[['home', 'draw', 'away']]
    
    naive_brier = compute_brier_score(y_true_dummies, naive_probs)
    mejora = (naive_brier - wf_results['aggregate_brier']) / naive_brier
    
    logger.info(
        "Comparación Brier Score",
        modelo=wf_results['aggregate_brier'],
        baseline_naive=naive_brier,
        mejora=f"{mejora:.2%}"
    )

    # Registrar métricas globales en MLflow
    tracker.log_metrics({
        "wf_aggregate_brier": wf_results['aggregate_brier'],
        "wf_aggregate_logloss": wf_results['aggregate_logloss'],
        "wf_aggregate_accuracy": wf_results['aggregate_accuracy'],
        "baseline_naive_brier": naive_brier,
        "edge_vs_naive_pct": mejora * 100
    })
    
    # Registrar métricas individuales por fold
    for fold in wf_results['fold_metrics']:
        tracker.log_metrics({
            f"fold_{fold['fold']}_brier": fold['brier_score'],
            f"fold_{fold['fold']}_logloss": fold['log_loss'],
            f"fold_{fold['fold']}_accuracy": fold['accuracy']
        })

    # 2.5 Ajustar y PERSISTIR el calibrador de probabilidades 1X2.
    # Se entrena con las predicciones out-of-fold del walk-forward (sin fuga de
    # información) y se guarda para que el dashboard de Value Bets calcule el EV
    # con probabilidades calibradas. Las λ crudas quedan intactas.
    calibrator = _fit_persist_calibrator(wf_results, tracker)

    # 3. Backtesting Financiero (Flat Stake + Kelly) sobre las cuotas reales.
    # Se cruza con las probabilidades CALIBRADAS (capa de evaluación financiera).
    backtest_results = _run_financial_backtest(
        wf_results=wf_results,
        engine=engine,
        tracker=tracker,
        calibrator=calibrator,
    )
    roi_backtest = 0.0
    if backtest_results:
        wf_results['backtest'] = backtest_results
        roi_backtest = backtest_results['flat_stake']['roi_pct']

    # 4. Persistir las métricas REALES en ML_MODEL_VERSION para el dashboard.
    # train() insertó la fila con brier/log_loss/ROI = 0.0 (esas métricas no se
    # producen durante el entrenamiento); aquí completamos la versión más
    # reciente con los valores del walk-forward, de modo que el dashboard de
    # Backtesting deje de mostrar ceros.
    tracker.update_latest_model_metrics(
        brier_score=wf_results['aggregate_brier'],
        log_loss=wf_results['aggregate_logloss'],
        roi_backtest=roi_backtest,
    )

    tracker.end_run()
    logger.info("Evaluación registrada exitosamente en MLflow.")

    return wf_results


def _fit_persist_calibrator(wf_results: dict, tracker) -> ProbabilityCalibrator | None:
    """Ajusta el calibrador 1X2 con las predicciones out-of-fold y lo persiste.

    Reporta la mejora del Brier multiclase RAW vs CALIBRADO estimada con
    validación cruzada de 5 folds sobre las predicciones OOF (mejora *fuera de
    muestra*, honesta), registra las métricas en MLflow y guarda el calibrador
    final (ajustado con todas las OOF) en el volumen del modelo.
    """
    predictions = wf_results.get("predictions", [])
    actuals = wf_results.get("actuals", [])
    if len(predictions) < 50:
        logger.warning(
            "Calibrador omitido: muy pocas predicciones OOF.",
            n=len(predictions),
        )
        return None

    raw = ProbabilityCalibrator._to_array(predictions)
    y = np.asarray([str(a) for a in actuals])

    brier_raw = brier_multiclass(raw, y)

    # Estimación fuera de muestra del Brier calibrado (KFold) para no reportar
    # una mejora inflada por evaluar sobre el set de ajuste del calibrador.
    cal_oof = np.zeros_like(raw)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in kf.split(raw):
        c = ProbabilityCalibrator(CALIBRATION_METHOD).fit(raw[tr_idx], y[tr_idx])
        cal_oof[te_idx] = c.predict(raw[te_idx])
    brier_cal = brier_multiclass(cal_oof, y)
    mejora = (brier_raw - brier_cal) / brier_raw if brier_raw else 0.0

    logger.info(
        "Calibración de probabilidades 1X2",
        metodo=CALIBRATION_METHOD,
        brier_raw=round(brier_raw, 4),
        brier_calibrado=round(brier_cal, 4),
        mejora=f"{mejora:.2%}",
    )
    tracker.log_metrics({
        "calib_brier_raw": brier_raw,
        "calib_brier_calibrated": brier_cal,
        "calib_improvement_pct": mejora * 100,
    })

    # Calibrador final: ajustado con TODAS las predicciones OOF y persistido.
    calibrator = ProbabilityCalibrator(CALIBRATION_METHOD).fit(raw, y)
    try:
        path = _calibrator_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        calibrator.save(path)
    except Exception as exc:
        logger.warning("No se pudo persistir el calibrador", error=str(exc))

    return calibrator


def _run_financial_backtest(
    wf_results: dict, engine, tracker, calibrator=None
) -> dict | None:
    """Ejecuta el backtesting financiero cruzando predicciones y cuotas.

    Alinea las predicciones del Walk-Forward con las cuotas históricas de
    ``FACT_ODDS`` mediante el ``match_id``, y calcula el ROI con las
    estrategias Flat Stake y Kelly fraccionario. Si se proporciona un
    ``calibrator``, las probabilidades 1X2 se calibran antes de calcular el EV
    (capa de evaluación financiera). Si no hay cuotas disponibles para los
    partidos evaluados, se omite el backtesting de forma segura.
    """
    if engine is None:
        from src.utils.db import get_engine
        engine = get_engine()

    odds_map = load_odds_map(engine)
    match_ids = wf_results.get('match_ids', [])
    predictions = wf_results.get('predictions', [])
    actuals = wf_results.get('actuals', [])

    if not odds_map or not match_ids:
        logger.warning(
            "Backtesting omitido: sin cuotas o sin match_ids alineables.",
            cuotas=len(odds_map),
            match_ids=len(match_ids),
        )
        return None

    # Alinear predicción / cuota / resultado real por match_id. Si hay
    # calibrador, se sustituye la probabilidad cruda por la calibrada.
    aligned_preds: list[dict] = []
    aligned_odds: list[dict] = []
    aligned_actuals: list[str] = []
    for mid, pred, actual in zip(match_ids, predictions, actuals):
        odds = odds_map.get(int(mid)) if mid is not None else None
        if odds is None:
            continue
        if calibrator is not None and calibrator.fitted:
            ch, cd, ca = calibrator.predict_one(
                pred["home"], pred["draw"], pred["away"]
            )
            pred = {"home": ch, "draw": cd, "away": ca}
        aligned_preds.append(pred)
        aligned_odds.append(odds)
        aligned_actuals.append(actual)

    if not aligned_odds:
        logger.warning(
            "Backtesting omitido: ningún partido de test tiene cuotas."
        )
        return None

    logger.info(
        "Ejecutando backtesting financiero",
        partidos_alineados=len(aligned_odds),
    )

    backtester = FinancialBacktester()
    flat_roi = backtester.flat_stake_roi(aligned_preds, aligned_odds, aligned_actuals)
    kelly_roi = backtester.kelly_criterion_roi(aligned_preds, aligned_odds, aligned_actuals)

    tracker.log_metrics({
        "roi_flat_stake_pct": flat_roi['roi_pct'],
        "flat_total_bets": float(flat_roi['total_bets']),
        "flat_win_rate": flat_roi['win_rate'],
        "roi_kelly_pct": kelly_roi['roi_pct'],
        "kelly_final_bankroll": kelly_roi['final_bankroll'],
        "kelly_max_drawdown_pct": kelly_roi['max_drawdown'],
    })

    logger.info(
        "Backtesting financiero completado",
        roi_flat_pct=round(flat_roi['roi_pct'], 2),
        roi_kelly_pct=round(kelly_roi['roi_pct'], 2),
        apuestas_flat=flat_roi['total_bets'],
    )

    return {"flat_stake": flat_roi, "kelly": kelly_roi}

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.db import get_engine
    from src.features.feature_engineering import FeatureEngineer
    from src.utils.constants import FEATURE_COLUMNS, TARGET_HOME, TARGET_AWAY

    setup_logging(dev_mode=True)
    logger.info("Iniciando evaluación offline desde CLI")

    try:
        # Cargar datos desde el feature store
        engine = get_engine()
        fe = FeatureEngineer(engine=engine)
        full_data = fe.load_feature_store()

        if full_data.empty:
            logger.error(
                "No hay datos en el feature store. "
                "Ejecute la ingesta primero."
            )
            raise SystemExit(1)

        # Definir splits temporales para Walk-Forward
        # Se usa split por año: cada año es un fold
        full_data = full_data.sort_values('match_date').reset_index(drop=True)
        years = full_data['match_date'].dt.year.unique()

        splits = []
        for i, year in enumerate(sorted(years)):
            if i < 2:  # Necesitamos al menos 2 años de historia
                continue
            # Splits por fecha (formato que consume WalkForwardEvaluator):
            # se entrena con todo lo anterior al año de test y se evalúa
            # sobre el año completo, sin filtración de información futura.
            splits.append({
                'fold': i - 1,
                'train_end': pd.Timestamp(year=int(year) - 1, month=12, day=31),
                'test_start': pd.Timestamp(year=int(year), month=1, day=1),
                'test_end': pd.Timestamp(year=int(year), month=12, day=31),
                'test_year': int(year),
            })

        if not splits:
            logger.error(
                "No hay suficientes años de datos para Walk-Forward. "
                "Se requieren al menos 3 años."
            )
            raise SystemExit(1)

        logger.info(
            "Walk-Forward configurado",
            n_folds=len(splits),
            años=list(sorted(years)),
        )

        # Determinar columnas de features disponibles
        available_features = [
            col for col in FEATURE_COLUMNS if col in full_data.columns
        ]

        results = run_offline_evaluation(
            splits=splits,
            full_data=full_data,
            feature_cols=available_features,
            engine=engine,
        )

        logger.info(
            "Evaluación completada",
            brier=results.get('aggregate_brier', 'N/A'),
            logloss=results.get('aggregate_logloss', 'N/A'),
            accuracy=results.get('aggregate_accuracy', 'N/A'),
        )

    except SystemExit:
        raise
    except Exception as e:
        logger.error("Error fatal en evaluación offline", error=str(e))
        raise
