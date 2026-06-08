"""
Scraper de eloratings.net.
Extrae los ratings Elo históricos de las selecciones nacionales.
"""
import pandas as pd
import requests
import structlog
from datetime import datetime

from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)

class EloScraper(BaseScraper):
    """Obtiene ratings Elo desde Eloratings.net."""

    def _fetch(self, date: str = None) -> pd.DataFrame:
        """
        Trae los Elo ratings para una fecha específica o el histórico.
        """
        try:
            # Endpoint real de Eloratings para obtener los datos en TSV
            # Se usa World.tsv como ejemplo general (en la práctica se puede consultar por fecha)
            url = "https://www.eloratings.net/World.tsv"
            # response = requests.get(url)
            # response.raise_for_status()
            
            # TODO: Parsear el TSV crudo a un DataFrame con pandas de acuerdo a la fecha.
            # Este es un esquema del DataFrame que se debe devolver:
            data = {
                "team": ["Argentina", "France", "Brazil"],
                "elo_rating": [2143.0, 2122.0, 2095.0],
                "rating_date": [datetime.now().date()] * 3
            }
            df = pd.DataFrame(data)
                
            return df
        except Exception as e:
            logger.error("Error scrapeando eloratings.net", error=str(e))
            raise
