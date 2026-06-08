"""
Tests exhaustivos para los validadores Pydantic.
Asegura el cumplimiento de la Regla de Oro (no nulos -> centinelas).
"""
import pytest
from datetime import date
import math

from src.ingestion.validators import (
    TeamValidator, MatchValidator, EloHistoryValidator,
    FifaRatingValidator, MatchXGValidator, OddsValidator
)
from src.utils.constants import SENTINEL_INT, SENTINEL_FLOAT, SENTINEL_STR, SENTINEL_DATE

def test_team_validator_valid_data(sample_team_data):
    """Test datos válidos para equipo."""
    team = TeamValidator(**sample_team_data)
    assert team.team_name == 'Argentina'
    assert team.fifa_code == 'ARG'
    assert team.confederation == 'CONMEBOL'
    assert team.fifa_ranking == 1

def test_team_validator_null_handling():
    """Test manejo de nulos (None o NaN) convirtiéndolos a centinelas."""
    team = TeamValidator(team_name=None, fifa_code='UNK', confederation=float('nan'), fifa_ranking=None)
    assert team.team_name == SENTINEL_STR
    assert team.confederation == SENTINEL_STR
    assert team.fifa_ranking == SENTINEL_INT

def test_match_validator_valid_data(sample_match_data):
    """Test datos válidos para partido."""
    match = MatchValidator(**sample_match_data)
    assert match.competition_id == 1
    assert match.home_goals == 3
    assert match.is_knockout is True

def test_match_validator_null_handling():
    """Test manejo de nulos en partido."""
    match = MatchValidator(
        competition_id=1,
        home_team_id=1,
        away_team_id=2,
        home_goals=None,
        away_goals=float('nan'),
        venue=None,
        match_date=None,
        attendance=None
    )
    assert match.home_goals == 0 # Default para goles nulos es 0
    assert match.away_goals == 0
    assert match.venue == SENTINEL_STR
    assert match.match_date == SENTINEL_DATE
    assert match.attendance == SENTINEL_INT

def test_elo_validator_valid_data():
    """Test datos válidos para Elo."""
    elo = EloHistoryValidator(
        team_id=1, rating_date=date(2022, 1, 1), elo_rating=2100.5, elo_delta=15.2
    )
    assert elo.elo_rating == 2100.5

def test_fifa_rating_validator_null_handling():
    """Test nulos en FifaRating."""
    sv = FifaRatingValidator(
        team_id=1,
        attack_rating=None,
        overall_rating=float('nan'),
        midfield_rating=None,
        defence_rating=None
    )
    assert sv.attack_rating == SENTINEL_FLOAT
    assert sv.overall_rating == SENTINEL_FLOAT
    assert sv.midfield_rating == SENTINEL_FLOAT
    assert sv.defence_rating == SENTINEL_FLOAT

def test_xg_validator_valid_data():
    """Test datos válidos para xG."""
    xg = MatchXGValidator(
        match_id=1, home_xg=1.5, away_xg=0.8, home_xga=0.8, away_xga=1.5, home_npxg=1.5, away_npxg=0.8
    )
    assert xg.home_xg == 1.5

def test_odds_validator_valid_data():
    """Test datos válidos para Odds."""
    odds = OddsValidator(
        match_id=1, bookmaker='Bet365', odds_home=2.1, odds_draw=3.4, odds_away=3.5, odds_over25=1.9
    )
    assert odds.odds_home == 2.1

def test_all_validators_to_db_dict(sample_team_data):
    """Test método to_db_dict() exporta correctamente a dict para SQLAlchemy."""
    team = TeamValidator(**sample_team_data)
    db_dict = team.to_db_dict()
    assert isinstance(db_dict, dict)
    assert db_dict['team_name'] == 'Argentina'
    assert 'fifa_code' in db_dict
