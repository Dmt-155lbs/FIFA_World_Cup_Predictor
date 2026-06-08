"""
Scraper de SoFIFA usando soccerdata.
Obtiene los ratings FIFA (ataque, medio, defensa, overall) de las selecciones.
"""
import pandas as pd
import soccerdata as sd
import structlog

from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)

class SoFIFAScraper(BaseScraper):
    """Extrae ratings FIFA de selecciones desde SoFIFA."""

    def _fetch(self, league: str = 'INT-World Cup') -> pd.DataFrame:
        try:
            sofifa = sd.SoFIFA(leagues=[league], versions="latest")
            df = sofifa.read_team_ratings()
            return df.reset_index()
        except Exception as e:
            logger.error("Error scrapeando SoFIFA", error=str(e))
            raise
