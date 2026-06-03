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
