"""
Tests para la capa macro (XGBoost).
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.models.xgboost_trainer import XGBoostTrainer

@pytest.fixture
def synthetic_data():
    """Genera datos sintéticos para entrenar."""
    np.random.seed(42)
    n_samples = 100
    
    # 10 features aleatorios
    X = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f'feature_{i}' for i in range(10)])
    
    # Targets sintéticos poisson (dependientes de features)
    lambda_home = np.exp(X['feature_0'] * 0.5 + 1.0)
    lambda_away = np.exp(X['feature_1'] * 0.3 + 0.5)
    
    y_home = pd.Series(np.random.poisson(lambda_home))
    y_away = pd.Series(np.random.poisson(lambda_away))
    
    return X, y_home, y_away

def test_train_returns_metrics(synthetic_data):
    """Verifica que el entrenamiento funcione y devuelva métricas."""
    X, y_home, y_away = synthetic_data
    trainer = XGBoostTrainer(params={'n_estimators': 10})
    metrics = trainer.train(X, y_home, y_away, eval_fraction=0.2)
    
    assert 'eval_poisson_nloglik_home' in metrics
    assert 'eval_poisson_nloglik_away' in metrics
    assert metrics['eval_poisson_nloglik_home'] > 0
    assert metrics['eval_poisson_nloglik_away'] > 0

def test_predict_returns_positive_lambdas(synthetic_data):
    """Verifica que los lambdas predichos sean siempre > 0."""
    X, y_home, y_away = synthetic_data
    trainer = XGBoostTrainer(params={'n_estimators': 10})
    trainer.train(X, y_home, y_away)
    
    lambda_home, lambda_away = trainer.predict(X)
    
    assert isinstance(lambda_home, np.ndarray)
    assert isinstance(lambda_away, np.ndarray)
    assert (lambda_home >= 0.01).all()
    assert (lambda_away >= 0.01).all()

def test_save_load_roundtrip(synthetic_data):
    """Verifica serialización del modelo."""
    X, y_home, y_away = synthetic_data
    trainer = XGBoostTrainer(params={'n_estimators': 10})
    trainer.train(X, y_home, y_away)
    
    preds_home_orig, preds_away_orig = trainer.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "model")
        trainer.save(save_path)
        
        new_trainer = XGBoostTrainer()
        new_trainer.load(save_path)
        
        preds_home_new, preds_away_new = new_trainer.predict(X)
        
        np.testing.assert_array_almost_equal(preds_home_orig, preds_home_new)
        np.testing.assert_array_almost_equal(preds_away_orig, preds_away_new)
