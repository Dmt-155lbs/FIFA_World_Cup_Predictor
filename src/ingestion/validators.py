"""
Modelos Pydantic para validar los datos scrapeados y garantizar la Regla de Oro
(no valores nulos) antes de la inserción en la base de datos SQL Server.
"""
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from datetime import date
from typing import Optional, Any
import math

from src.utils.constants import SENTINEL_INT, SENTINEL_FLOAT, SENTINEL_STR, SENTINEL_DATE


class TeamValidator(BaseModel):
    team_name: str = SENTINEL_STR
    fifa_code: str = 'UNK'
    confederation: str = SENTINEL_STR
    fifa_ranking: int = SENTINEL_INT

    @field_validator('*', mode='before')
    def handle_nulls(cls, v: Any, info) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if info.field_name in ['fifa_ranking']:
                return SENTINEL_INT
            return SENTINEL_STR
        return v

    model_config = ConfigDict(from_attributes=True)

    def to_db_dict(self) -> dict:
        return self.model_dump()


class MatchValidator(BaseModel):
    competition_id: int
    match_date: date = SENTINEL_DATE
    home_team_id: int
    away_team_id: int
    home_goals: int = 0
    away_goals: int = 0
    venue: str = SENTINEL_STR
    is_neutral: bool = False
    is_knockout: bool = False
    attendance: int = SENTINEL_INT

    @field_validator('home_goals', 'away_goals', 'attendance', mode='before')
    def handle_int_nulls(cls, v: Any, info) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_INT if info.field_name == 'attendance' else 0
        return int(v)

    @field_validator('venue', mode='before')
    def handle_str_nulls(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_STR
        return str(v)

    @field_validator('match_date', mode='before')
    def handle_date_nulls(cls, v: Any) -> Any:
        if v is None:
            return SENTINEL_DATE
        return v

    model_config = ConfigDict(from_attributes=True)

    def to_db_dict(self) -> dict:
        return self.model_dump()


class EloHistoryValidator(BaseModel):
    team_id: int
    rating_date: date = SENTINEL_DATE
    elo_rating: float = SENTINEL_FLOAT
    elo_delta: float = SENTINEL_FLOAT
    
    @field_validator('elo_rating', 'elo_delta', mode='before')
    def handle_float_nulls(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_FLOAT
        return float(v)
        
    @field_validator('rating_date', mode='before')
    def handle_date_nulls(cls, v: Any) -> Any:
        if v is None:
            return SENTINEL_DATE
        return v
        
    model_config = ConfigDict(from_attributes=True)

    def to_db_dict(self) -> dict:
        return self.model_dump()


class SquadValueValidator(BaseModel):
    team_id: int
    valuation_date: date = SENTINEL_DATE
    market_value_eur: float = SENTINEL_FLOAT
    squad_size: int = SENTINEL_INT
    avg_age: float = SENTINEL_FLOAT
    total_caps: int = SENTINEL_INT
    total_minutes_season: int = SENTINEL_INT

    @field_validator('market_value_eur', 'avg_age', mode='before')
    def handle_float_nulls(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_FLOAT
        return float(v)

    @field_validator('squad_size', 'total_caps', 'total_minutes_season', mode='before')
    def handle_int_nulls(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_INT
        return int(v)
        
    @field_validator('valuation_date', mode='before')
    def handle_date_nulls(cls, v: Any) -> Any:
        if v is None:
            return SENTINEL_DATE
        return v

    model_config = ConfigDict(from_attributes=True)

    def to_db_dict(self) -> dict:
        return self.model_dump()


class MatchXGValidator(BaseModel):
    match_id: int
    home_xg: float = SENTINEL_FLOAT
    away_xg: float = SENTINEL_FLOAT
    home_xga: float = SENTINEL_FLOAT
    away_xga: float = SENTINEL_FLOAT
    home_npxg: float = SENTINEL_FLOAT
    away_npxg: float = SENTINEL_FLOAT

    @field_validator('*', mode='before')
    def handle_float_nulls(cls, v: Any, info) -> Any:
        if info.field_name == 'match_id':
            return v
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_FLOAT
        return float(v)

    model_config = ConfigDict(from_attributes=True)

    def to_db_dict(self) -> dict:
        return self.model_dump()


class OddsValidator(BaseModel):
    match_id: int
    bookmaker: str = SENTINEL_STR
    odds_home: float = SENTINEL_FLOAT
    odds_draw: float = SENTINEL_FLOAT
    odds_away: float = SENTINEL_FLOAT
    odds_over25: float = SENTINEL_FLOAT

    @field_validator('odds_home', 'odds_draw', 'odds_away', 'odds_over25', mode='before')
    def handle_float_nulls(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_FLOAT
        return float(v)
        
    @field_validator('bookmaker', mode='before')
    def handle_str_nulls(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return SENTINEL_STR
        return str(v)

    model_config = ConfigDict(from_attributes=True)

    def to_db_dict(self) -> dict:
        return self.model_dump()
