import pandas as pd
import numpy as np
from src.evaluation.metrics import compute_brier_score, compute_log_loss, compute_accuracy
from src.evaluation.financial_backtest import FinancialBacktester

def test_metrics():
    y_true_dummies = pd.DataFrame({'home': [1, 0, 0], 'draw': [0, 1, 0], 'away': [0, 0, 1]})
    y_pred_probs = pd.DataFrame({'home': [0.8, 0.1, 0.1], 'draw': [0.1, 0.7, 0.2], 'away': [0.1, 0.2, 0.7]})
    y_true = pd.Series(['home', 'draw', 'away'])

    brier = compute_brier_score(y_true_dummies, y_pred_probs)
    logloss = compute_log_loss(y_true, y_pred_probs)
    acc = compute_accuracy(y_true, y_pred_probs)

    assert brier >= 0
    assert logloss >= 0
    assert acc == 1.0

def test_financial_backtester():
    predictions = [{'home': 0.5, 'draw': 0.3, 'away': 0.2}, {'home': 0.2, 'draw': 0.2, 'away': 0.6}]
    odds = [{'home': 2.5, 'draw': 3.0, 'away': 4.0}, {'home': 4.0, 'draw': 3.0, 'away': 1.8}]
    actuals = ['home', 'away']

    backtester = FinancialBacktester()
    
    flat_res = backtester.flat_stake_roi(predictions, odds, actuals, stake=10, ev_threshold=0.0)
    assert flat_res['total_bets'] > 0
    assert flat_res['roi_pct'] > 0

    kelly_res = backtester.kelly_criterion_roi(predictions, odds, actuals, fraction=0.25, initial_bankroll=100)
    assert kelly_res['total_bets'] > 0
