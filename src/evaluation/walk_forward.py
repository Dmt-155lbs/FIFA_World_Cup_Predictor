import pandas as pd
import numpy as np
import structlog
from typing import Any
from src.evaluation.metrics import compute_brier_score, compute_log_loss, compute_accuracy

logger = structlog.get_logger(__name__)

class WalkForwardEvaluator:
    """
    Evaluación Walk-Forward estricta:
    - La ventana de entrenamiento se expande cronológicamente
    - NUNCA se usa información futura para entrenar
    - Se evalúa en la siguiente ventana temporal
    """

    def __init__(self, splits: list[dict]):
        """
        splits = [
            {'train_end': '2021-12-31', 'test_start': '2022-11-20', 'test_end': '2022-12-18'},
            {'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'},
            ...
        ]
        """
        self.splits = splits

    def evaluate(self, pipeline: Any, full_data: pd.DataFrame, feature_cols: list[str]) -> dict:
        all_predictions = []
        all_actuals = []
        fold_metrics = []

        for i, split in enumerate(self.splits):
            logger.info("Evaluando fold.", fold=i, split=split)
            # Separar estrictamente por fecha
            train = full_data[full_data['match_date'] <= split['train_end']]
            test = full_data[
                (full_data['match_date'] >= split['test_start']) & 
                (full_data['match_date'] <= split['test_end'])
            ]
            
            if len(train) == 0 or len(test) == 0:
                logger.warning("Train o Test vacío. Saltando fold.", fold=i)
                continue

            X_train = train[feature_cols]
            y_train_home = train['home_goals']
            y_train_away = train['away_goals']

            # Entrenar en ventana histórica
            pipeline.train(X=X_train, y_home=y_train_home, y_away=y_train_away, optimize_hyperparams=False)

            preds_list = []
            for idx, row in test.iterrows():
                row_df = pd.DataFrame([row[feature_cols]])
                pred = pipeline.predict_match(row_df, home_team="Home", away_team="Away")
                preds_list.append({
                    'home': pred.prob_home,
                    'draw': pred.prob_draw,
                    'away': pred.prob_away
                })
            
            preds_df = pd.DataFrame(preds_list)
            
            # Crear true dummies para métricas
            test_outcome = test.apply(lambda x: 'home' if x['home_goals'] > x['away_goals'] else ('away' if x['away_goals'] > x['home_goals'] else 'draw'), axis=1)
            
            y_true_dummies = pd.get_dummies(test_outcome)
            for col in ['home', 'draw', 'away']:
                if col not in y_true_dummies.columns:
                    y_true_dummies[col] = 0
            y_true_dummies = y_true_dummies[['home', 'draw', 'away']]

            # Métricas del fold
            brier = compute_brier_score(y_true_dummies, preds_df)
            logloss = compute_log_loss(test_outcome, preds_df)
            acc = compute_accuracy(test_outcome, preds_df)

            fold_metric = {
                'fold': i,
                'train_size': len(train),
                'test_size': len(test),
                'brier_score': brier,
                'log_loss': logloss,
                'accuracy': acc,
            }
            logger.info("Fold metrics", **fold_metric)
            fold_metrics.append(fold_metric)
            
            all_predictions.extend(preds_list)
            all_actuals.extend(test_outcome.values)

        return {
            'fold_metrics': fold_metrics,
            'aggregate_brier': np.mean([f['brier_score'] for f in fold_metrics]) if fold_metrics else 0,
            'aggregate_logloss': np.mean([f['log_loss'] for f in fold_metrics]) if fold_metrics else 0,
            'aggregate_accuracy': np.mean([f['accuracy'] for f in fold_metrics]) if fold_metrics else 0,
            'predictions': all_predictions,
            'actuals': all_actuals
        }
