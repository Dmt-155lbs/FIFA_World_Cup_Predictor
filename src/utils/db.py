"""
Gestión de conexiones a la base de datos SQL Server mediante SQLAlchemy 2.0.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import logging

from src.config import get_settings

logger = logging.getLogger(__name__)

# Singleton engine y SessionLocal
_engine = None
_SessionLocal = None

def get_engine():
    """Retorna o inicializa el motor SQLAlchemy con connection pooling."""
    global _engine
    if _engine is None:
        settings = get_settings()
        kwargs = {"pool_pre_ping": True}
        if not settings.db_connection_string.startswith("sqlite"):
            kwargs["pool_size"] = 5
            kwargs["max_overflow"] = 10
            
        _engine = create_engine(
            settings.db_connection_string,
            **kwargs
        )
    return _engine

def get_session_factory():
    """Retorna o inicializa la fábrica de sesiones."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager para transacciones de base de datos.
    Asegura commit en éxito y rollback en error.
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error en transacción de BD: {e}")
        raise
    finally:
        session.close()

def test_connection() -> bool:
    """Verifica si la conexión a la base de datos es exitosa ejecutando un SELECT 1."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Fallo al conectar a la BD: {e}")
        return False
