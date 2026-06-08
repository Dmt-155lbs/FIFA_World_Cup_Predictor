"""
Constantes globales del proyecto.
Incluye centinelas para el manejo de valores nulos (Regla de Oro de la BD).
"""
from datetime import date

# Centinelas NOT NULL (deben usarse antes de insertar en BD)
SENTINEL_INT = -1
SENTINEL_FLOAT = 0.0
SENTINEL_STR = 'UNKNOWN'
SENTINEL_DATE = date(1900, 1, 1)

# K-Factors Elo (ajustes según importancia del partido)
K_FACTOR = {
    'world_cup': 40,
    'continental_cup': 35,
    'qualifier': 30,
    'nations_league': 30,
    'friendly': 20,
}

# Confederaciones FIFA
CONFEDERATIONS = ['UEFA', 'CONMEBOL', 'CONCACAF', 'CAF', 'AFC', 'OFC']

# Parámetros del Modelo
MAX_GOALS_MATRIX = 8
EXTRA_TIME_FACTOR = 1/3
PENALTY_ELO_BONUS = 0.03
FATIGUE_THRESHOLD_MINUTES = 4500
ROLLING_WINDOW_MATCHES = 10
ELO_MOMENTUM_MONTHS = 6

# Fuentes de datos
DATA_START_YEAR = 2014
FOOTBALL_DATA_BASE_URL = 'https://www.football-data.co.uk'
ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'

# ============================================================================
# Constantes del Ensemble Híbrido (Fase 2)
# ============================================================================

# Columnas de features para XGBoost (orden estricto)
FEATURE_COLUMNS = [
    # Elo
    'home_elo', 'away_elo', 'elo_diff',
    'home_elo_delta', 'away_elo_delta',
    # xG
    'home_xg', 'away_xg',
    'home_xga', 'away_xga',
    'home_npxg', 'away_npxg',
    # Valor de plantilla
    'home_fifa_attack', 'away_fifa_attack',
    'home_fifa_overall', 'away_fifa_overall',
    'home_fifa_midfield', 'away_fifa_midfield',
    'home_fifa_defence', 'away_fifa_defence',
    # Contexto del partido
    'is_neutral', 'is_knockout',
    'competition_weight',
    # Rolling features (calculados en FeatureEngineer)
    'home_rolling_goals_scored', 'away_rolling_goals_scored',
    'home_rolling_goals_conceded', 'away_rolling_goals_conceded',
    'home_rolling_xg', 'away_rolling_xg',
    'home_rolling_xga', 'away_rolling_xga',
    'home_rolling_form', 'away_rolling_form',
    'home_elo_momentum', 'away_elo_momentum',
    # Diferenciales
    'xg_diff', 'form_diff', 'fifa_attack_diff',
    'goals_diff',
    # Contexto temporal
    'home_days_rest', 'away_days_rest',
]

# Nombres de las variables target
TARGET_HOME = 'home_goals'
TARGET_AWAY = 'away_goals'

# Peso normalizado de competiciones (importancia relativa, 0–1)
COMPETITION_WEIGHTS = {k: v / 40.0 for k, v in K_FACTOR.items()}

# Umbral para apuestas over/under
OVER_UNDER_THRESHOLD = 2.5

# Columnas a excluir del feature set (metadatos, no predictivas)
DROP_COLUMNS = [
    'match_id', 'match_date', 'competition_id', 'competition_name', 'stage',
    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
    'home_fifa_code', 'away_fifa_code', 'venue', 'attendance',
    'ingested_at', 'target_result',
]

