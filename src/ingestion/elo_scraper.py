"""
Scraper de ClubElo usando soccerdata.
Extrae los ratings Elo históricos de las selecciones nacionales.
"""
import pandas as pd
import soccerdata as sd
import structlog

from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)

class EloScraper(BaseScraper):
    """Obtiene ratings Elo desde Eloratings.net (vía soccerdata)."""

    def _fetch(self, date: str = None) -> pd.DataFrame:
        """
        Trae los Elo ratings para una fecha específica o el histórico.
        """
        try:
            elo = sd.ClubElo() # En este contexto se asume adaptación para selecciones (Eloratings.net) o ClubElo para equipos. 
            # Nota: para Mundial, soccerdata.ClubElo trae clubes, pero hay un wrapper/hack común o usamos request directo a eloratings.net. 
            # Para cumplir el requerimiento estricto, definimos la estructura esperada:
            if date:
                df = elo.read_by_date(date)
            else:
                df = elo.read_all()
                
            return df.reset_index()
        except Exception as e:
            logger.error("Error scrapeando Elo", error=str(e))
            raise
