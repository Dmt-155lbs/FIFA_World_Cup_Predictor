"""
Clase base para los scrapers con lógica de tolerancia a fallos.
"""
from abc import ABC, abstractmethod
from typing import Optional
import os
import pickle
import time
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog
from pathlib import Path

from src.config import get_settings

logger = structlog.get_logger(__name__)

class BaseScraper(ABC):
    """
    Scraper base con circuit breaker, retry exponencial y fallback a caché.
    """
    MAX_RETRIES = 3
    
    def __init__(self):
        settings = get_settings()
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_time = 3  # Rate limiting base

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} para scraping."
        )
    )
    def fetch_with_retry(self, **kwargs) -> pd.DataFrame:
        """Intenta ejecutar el scraping con reintentos exponenciales."""
        time.sleep(self.sleep_time) # Rate limit simple
        return self._fetch(**kwargs)

    @abstractmethod
    def _fetch(self, **kwargs) -> pd.DataFrame:
        """Implementación específica de cada fuente de datos (FBref, Elo, etc)."""
        pass

    def fetch_or_cache(self, cache_key: str, **kwargs) -> pd.DataFrame:
        """
        Estrategia robusta: Live scraping → Cache local.
        Si falla en live, carga la última versión de caché y alerta con warning.
        """
        try:
            data = self.fetch_with_retry(**kwargs)
            self._save_cache(cache_key, data)
            return data
        except Exception as e:
            logger.warning("Scraping fallido. Usando caché.", cache_key=cache_key, error=str(e))
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached
            raise RuntimeError(f"Error fatal: Sin datos vivos ni caché para '{cache_key}'") from e

    def _save_cache(self, key: str, data: pd.DataFrame) -> None:
        """Guarda dataframe en caché local (pickle para tipos complejos)."""
        filepath = self.cache_dir / f"{key}.pkl"
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error("Error guardando caché", key=key, error=str(e))

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Carga dataframe desde caché local."""
        filepath = self.cache_dir / f"{key}.pkl"
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error("Error cargando caché", key=key, error=str(e))
        return None
