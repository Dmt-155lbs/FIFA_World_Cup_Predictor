import pandas as pd
import structlog
from src.pipeline import PredictionPipeline
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.financial_backtest import FinancialBacktester
from src.evaluation.metrics import generate_naive_baseline, compute_brier_score
from src.models.mlflow_tracking import ExperimentTracker

logger = structlog.get_logger(__name__)

def run_offline_evaluation(splits: list[dict], full_data: pd.DataFrame, feature_cols: list[str]):
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

    # 3. (Opcional) Backtesting Financiero si se tienen cuotas
    # backtester = FinancialBacktester()
    # odds = ... # Cargar odds del dataset de test global
    # flat_roi = backtester.flat_stake_roi(wf_results['predictions'], odds, wf_results['actuals'])
    # kelly_roi = backtester.kelly_criterion_roi(wf_results['predictions'], odds, wf_results['actuals'])
    # tracker.log_metrics({
    #     "roi_flat_stake": flat_roi['roi_pct'],
    #     "roi_kelly": kelly_roi['roi_pct']
    # })
    
    tracker.end_run()
    logger.info("Evaluación registrada exitosamente en MLflow.")
    
    return wf_results

if __name__ == "__main__":
    # Aquí iría el código de inicialización para cargar datos de DB y definir splits
    pass
