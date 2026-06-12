import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import structlog

logger = structlog.get_logger(__name__)

def compute_brier_score(y_true_dummies: pd.DataFrame, y_pred_probs: pd.DataFrame) -> float:
    """
    Computes Brier Score for multi-class classification.
    y_true_dummies: DataFrame with binary indicators for 'home', 'draw', 'away'
    y_pred_probs: DataFrame with predicted probabilities for 'home', 'draw', 'away'
    """
    brier_sum = 0.0
    for col in ['home', 'draw', 'away']:
        # .values fuerza la resta posicional: los dos DataFrames pueden llegar
        # con índices distintos (preds con RangeIndex, dummies con el índice
        # original del test), y restar Series desalineadas produciría NaN.
        brier_sum += np.mean(
            (y_pred_probs[col].values - y_true_dummies[col].values) ** 2
        )
    return brier_sum / 3.0

def compute_log_loss(y_true: pd.Series, y_pred_probs: pd.DataFrame) -> float:
    """
    Computes Log Loss.
    """
    return log_loss(y_true, y_pred_probs[['home', 'draw', 'away']])

def compute_accuracy(y_true: pd.Series, y_pred_probs: pd.DataFrame) -> float:
    """
    Computes Accuracy (1X2).
    """
    predictions = y_pred_probs[['home', 'draw', 'away']].idxmax(axis=1)
    return accuracy_score(y_true, predictions)

def plot_calibration_curve(y_true: pd.Series, y_pred_probs: pd.DataFrame, save_path: str = "calibration_curve.png"):
    """
    Plots and saves the calibration curve for home, draw, away probabilities.
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for outcome in ['home', 'draw', 'away']:
        y_true_binary = (y_true == outcome).astype(int)
        prob_pos, prob_pred = calibration_curve(y_true_binary, y_pred_probs[outcome], n_bins=10)
        
        ax1.plot(prob_pred, prob_pos, "s-", label=f"{outcome.capitalize()}")
        ax2.hist(y_pred_probs[outcome], range=(0, 1), bins=10, label=outcome, histtype="step", lw=2)
        
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots (reliability curve)")
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Calibration curve saved.", path=save_path)

def generate_naive_baseline(n_samples: int) -> pd.DataFrame:
    """
    Genera probabilidades base ingenuas (33.3% para cada resultado).
    Sirve como baseline inferior.
    """
    return pd.DataFrame({
        'home': [1/3] * n_samples,
        'draw': [1/3] * n_samples,
        'away': [1/3] * n_samples
    })

def generate_implicit_odds_baseline(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las cuotas decimales a probabilidades implícitas, 
    eliminando el margen de la casa de apuestas (overround).
    odds_df: DataFrame con columnas ['odds_home', 'odds_draw', 'odds_away']
    """
    impl_home = 1.0 / odds_df['odds_home']
    impl_draw = 1.0 / odds_df['odds_draw']
    impl_away = 1.0 / odds_df['odds_away']
    
    # El overround (margen) hace que la suma sea > 1
    margin = impl_home + impl_draw + impl_away
    
    # Probabilidades reales estimadas por el mercado
    return pd.DataFrame({
        'home': impl_home / margin,
        'draw': impl_draw / margin,
        'away': impl_away / margin
    })

