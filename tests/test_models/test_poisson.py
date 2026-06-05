"""
Tests para la capa estocástica (Poisson Bivariado con Dixon-Coles).
"""
import pytest
import numpy as np

from src.models.poisson_model import BivariatePoisson

def test_score_matrix_sums_to_one():
    """La matriz de probabilidad debe sumar 1.0 (o estar muy cerca)."""
    model = BivariatePoisson(max_goals=8, rho=-0.13)
    matrix = model.score_matrix(1.5, 1.2)
    assert abs(matrix.sum() - 1.0) < 1e-6

def test_probabilities_sum_to_one():
    """La suma de prob_home, prob_draw, prob_away debe ser 1.0."""
    model = BivariatePoisson(max_goals=8, rho=-0.13)
    matrix = model.score_matrix(1.5, 1.2)
    probs = model.match_probabilities(matrix)
    
    total = probs['prob_home'] + probs['prob_draw'] + probs['prob_away']
    assert abs(total - 1.0) < 1e-6

def test_rho_zero_equals_independent():
    """Si rho es 0, el modelo equivale a dos distribuciones Poisson independientes."""
    model_indep = BivariatePoisson(max_goals=8, rho=0.0)
    model_dc = BivariatePoisson(max_goals=8, rho=-0.13)
    
    matrix_indep = model_indep.score_matrix(1.5, 1.2)
    matrix_dc = model_dc.score_matrix(1.5, 1.2)
    
    # La probabilidad de 0-0 debe ser mayor en DC debido a la corrección (rho negativo reduce la resta, o sea aumenta prob)
    # tau(0,0) = 1 - lambda1*lambda2*rho
    assert matrix_dc[0, 0] > matrix_indep[0, 0]

def test_over_under_complement():
    """Over + Under debe sumar 1.0."""
    model = BivariatePoisson(max_goals=8)
    matrix = model.score_matrix(1.5, 1.2)
    ou = model.over_under_probs(matrix, 2.5)
    
    assert abs(ou['prob_over'] + ou['prob_under'] - 1.0) < 1e-6

def test_expected_goals_reasonable():
    """Los goles esperados calculados desde la matriz deben aproximar los lambdas de entrada."""
    lambda_h, lambda_a = 1.5, 1.2
    model = BivariatePoisson(max_goals=15)  # Más grande para capturar la cola
    matrix = model.score_matrix(lambda_h, lambda_a, rho=0.0)
    
    eh, ea = model.expected_goals(matrix)
    assert abs(eh - lambda_h) < 0.05
    assert abs(ea - lambda_a) < 0.05

def test_most_likely_scores_ordered():
    """Los marcadores más probables deben estar ordenados descendentemente."""
    model = BivariatePoisson(max_goals=8)
    matrix = model.score_matrix(1.5, 1.2)
    scores = model.most_likely_scores(matrix, top_n=5)
    
    assert len(scores) == 5
    for i in range(1, len(scores)):
        assert scores[i-1][2] >= scores[i][2]
