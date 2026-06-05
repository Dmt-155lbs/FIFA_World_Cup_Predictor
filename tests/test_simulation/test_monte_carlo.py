"""
Tests para la capa estructural (BracketEngine y MonteCarloSimulator).
"""
import pytest
import numpy as np
import yaml
import tempfile
from pathlib import Path

from src.simulation.bracket_engine import BracketEngine
from src.simulation.monte_carlo import MonteCarloSimulator

@pytest.fixture
def dummy_bracket_config(tmp_path):
    """Crea un config YAML mínimo para pruebas."""
    config = {
        'torneo': {'total_grupos': 2, 'total_equipos': 8},
        'fase_de_grupos': {
            'formato': {'partidos_por_grupo': 6, 'puntos_victoria': 3, 'puntos_empate': 1, 'puntos_derrota': 0},
            'grupos': {
                'A': [
                    {'posicion': 1, 'equipo': 'Team_A1', 'fifa_code': 'A1'},
                    {'posicion': 2, 'equipo': 'Team_A2', 'fifa_code': 'A2'},
                    {'posicion': 3, 'equipo': 'Team_A3', 'fifa_code': 'A3'},
                    {'posicion': 4, 'equipo': 'Team_A4', 'fifa_code': 'A4'},
                ],
                'B': [
                    {'posicion': 1, 'equipo': 'Team_B1', 'fifa_code': 'B1'},
                    {'posicion': 2, 'equipo': 'Team_B2', 'fifa_code': 'B2'},
                    {'posicion': 3, 'equipo': 'Team_B3', 'fifa_code': 'B3'},
                    {'posicion': 4, 'equipo': 'Team_B4', 'fifa_code': 'B4'},
                ]
            }
        },
        'reglas_desempate': {
            'criterios': [
                {'criterio': 'puntos'},
                {'criterio': 'diferencia_de_goles'},
                {'criterio': 'goles_a_favor'}
            ]
        },
        'mejores_terceros': {
            'cantidad_clasificados': 0, # Simplificado para test
            'tabla_asignacion': {}
        },
        'bracket_eliminatorio': {
            'semifinales': [
                {'id': 'SF_01', 'local': '1A', 'visitante': '2B', 'ronda': 'Semifinal'},
                {'id': 'SF_02', 'local': '1B', 'visitante': '2A', 'ronda': 'Semifinal'},
            ],
            'tercer_puesto': [
                {'id': 'TP_01', 'local': 'L_SF_01', 'visitante': 'L_SF_02', 'ronda': 'Tercer Puesto'}
            ],
            'final': [
                {'id': 'FIN_01', 'local': 'W_SF_01', 'visitante': 'W_SF_02', 'ronda': 'Final'}
            ]
        },
        'parametros_eliminatoria': {
            'factor_tiempo_extra': 0.3333,
            'bonus_elo_penales': 0.03,
            'prob_penales_base': 0.25
        }
    }
    
    file_path = tmp_path / "dummy_bracket.yaml"
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    return str(file_path)

@pytest.fixture
def dummy_lambdas():
    lambdas = {}
    for g in ['A', 'B']:
        for i in range(1, 5):
            team = f'Team_{g}{i}'
            lambdas[team] = (1.5, 1.2) # lambda_home, lambda_away
    return lambdas

def test_bracket_engine_simulate_group_stage(dummy_bracket_config, dummy_lambdas):
    engine = BracketEngine(config_path=dummy_bracket_config)
    rng = np.random.default_rng(42)
    
    standings = engine.simulate_group_stage(dummy_lambdas, rng)
    
    assert 'A' in standings
    assert 'B' in standings
    assert len(standings['A']) == 4
    
    for team_stats in standings['A']:
        assert team_stats['partidos'] == 3

def test_bracket_engine_simulate_full_bracket(dummy_bracket_config, dummy_lambdas):
    engine = BracketEngine(config_path=dummy_bracket_config)
    rng = np.random.default_rng(42)
    
    results = engine.simulate_full_bracket(dummy_lambdas, rng)
    
    assert 'champion' in results
    assert 'finalist' in results
    assert 'third' in results
    assert results['champion'] != results['finalist']

def test_monte_carlo_simulate(dummy_bracket_config, dummy_lambdas):
    engine = BracketEngine(config_path=dummy_bracket_config)
    simulator = MonteCarloSimulator(bracket_engine=engine, seed=42)
    
    results = simulator.simulate(dummy_lambdas, n_simulations=10)
    
    assert results.n_simulations == 10
    # La suma de probabilidades de campeón debe ser ~1.0
    assert abs(sum(results.champion_probs.values()) - 1.0) < 1e-6
    
    # Todos los equipos deben estar en las probabilidades de fase de grupos
    assert len(results.group_stage_probs) == 8

def test_monte_carlo_reproducibility(dummy_bracket_config, dummy_lambdas):
    engine = BracketEngine(config_path=dummy_bracket_config)
    
    sim1 = MonteCarloSimulator(bracket_engine=engine, seed=123)
    res1 = sim1.simulate(dummy_lambdas, n_simulations=10)
    
    sim2 = MonteCarloSimulator(bracket_engine=engine, seed=123)
    res2 = sim2.simulate(dummy_lambdas, n_simulations=10)
    
    # Los resultados exactos deben coincidir si la semilla es la misma
    assert res1.champion_probs == res2.champion_probs
