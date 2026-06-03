"""
Tests de integración para la conexión a la base de datos y la sesión.
"""
from sqlalchemy import text
from unittest.mock import patch

from src.utils.db import get_engine, get_session, test_connection

def test_engine_creation(mock_db_engine):
    """Verifica que el engine se crea correctamente (usamos mock en pruebas)."""
    with patch('src.utils.db.get_engine', return_value=mock_db_engine):
        engine = get_engine()
        assert engine is not None
        assert str(engine.url) == 'sqlite:///:memory:'

def test_session_context_manager(mock_db_engine):
    """Verifica el manejo de la sesión transaccional."""
    with patch('src.utils.db.get_engine', return_value=mock_db_engine):
        with get_session() as session:
            # Ejecutar un comando simple en SQLite memory
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1

def test_connection_string_parsing():
    """Verifica que la prueba de conexión (SELECT 1) funciona."""
    with patch('src.utils.db.get_engine') as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.return_value = True
        
        is_connected = test_connection()
        assert is_connected is True
        mock_conn.execute.assert_called_once()
