"""
Fixtures compartidas de pytest.
"""
import pytest
from sqlalchemy import create_engine
from datetime import date
import tempfile
import os
import shutil

@pytest.fixture(scope="session")
def mock_db_engine():
    """Engine de SQLite en memoria para tests."""
    engine = create_engine('sqlite:///:memory:', pool_pre_ping=True)
    yield engine

@pytest.fixture
def sample_match_data():
    """Datos de un partido de ejemplo."""
    return {
        'competition_id': 1,
        'match_date': date(2022, 12, 18),
        'home_team_id': 1, # Argentina
        'away_team_id': 2, # France
        'home_goals': 3,
        'away_goals': 3,
        'venue': 'Lusail Stadium',
        'is_neutral': True,
        'is_knockout': True,
        'attendance': 88966
    }

@pytest.fixture
def sample_team_data():
    """Datos de un equipo de ejemplo."""
    return {
        'team_name': 'Argentina',
        'fifa_code': 'ARG',
        'confederation': 'CONMEBOL',
        'fifa_ranking': 1
    }

@pytest.fixture
def tmp_cache_dir():
    """Directorio temporal para tests de caché."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)
