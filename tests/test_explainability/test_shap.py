"""
Tests para el módulo de explicabilidad SHAP.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.explainability.shap_analysis import SHAPAnalyzer


@pytest.fixture
def synthetic_shap_data():
    """Genera modelos y datos sintéticos pequeños para pruebas de SHAP."""
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    feature_names = [f"feature_{i}" for i in range(n_features)]

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), columns=feature_names
    )
    
    # Objetivos dummy poisson
    y_home = np.random.poisson(1.5, size=n_samples)
    y_away = np.random.poisson(1.2, size=n_samples)

    # Entrenar modelos rápidos
    model_home = XGBRegressor(n_estimators=10, max_depth=3, objective="count:poisson")
    model_home.fit(X, y_home)

    model_away = XGBRegressor(n_estimators=10, max_depth=3, objective="count:poisson")
    model_away.fit(X, y_away)

    return model_home, model_away, X, feature_names


def test_shap_compute_values(synthetic_shap_data):
    """Verifica que se pueden calcular los SHAP values."""
    model_home, model_away, X, feature_names = synthetic_shap_data
    
    analyzer = SHAPAnalyzer(model_home, model_away, feature_names)
    shap_home, shap_away = analyzer.compute_shap_values(X)
    
    # Para XGBoost y TreeExplainer, shap_values retorna array de (n_samples, n_features)
    assert isinstance(shap_home, np.ndarray)
    assert isinstance(shap_away, np.ndarray)
    assert shap_home.shape == (len(X), len(feature_names))
    assert shap_away.shape == (len(X), len(feature_names))


def test_shap_plot_summary(synthetic_shap_data):
    """Verifica que se generen los archivos de summary plot."""
    model_home, model_away, X, feature_names = synthetic_shap_data
    analyzer = SHAPAnalyzer(model_home, model_away, feature_names)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_files = analyzer.plot_summary(X, save_dir=tmpdir)
        
        assert len(saved_files) == 2
        for f in saved_files:
            assert os.path.exists(f)
            assert f.endswith(".png")


def test_shap_explain_match(synthetic_shap_data):
    """Verifica que se generen los force/waterfall plots para un partido."""
    model_home, model_away, X, feature_names = synthetic_shap_data
    analyzer = SHAPAnalyzer(model_home, model_away, feature_names)
    
    single_match = X.iloc[[0]]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_files = analyzer.explain_match(
            match_features=single_match,
            team_home="TestHome",
            team_away="TestAway",
            save_dir=tmpdir
        )
        
        assert len(saved_files) == 2
        for f in saved_files:
            assert os.path.exists(f)
            assert f.endswith(".png")


def test_shap_explain_match_requires_single_row(synthetic_shap_data):
    """Verifica que explain_match falle si recibe múltiples filas."""
    model_home, model_away, X, feature_names = synthetic_shap_data
    analyzer = SHAPAnalyzer(model_home, model_away, feature_names)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="exactamente 1 fila"):
            analyzer.explain_match(
                match_features=X,  # Varias filas
                save_dir=tmpdir
            )
