"""
Scraper de TransferMarkt usando soccerdata.
Obtiene los valores de mercado de las plantillas.
"""
import pandas as pd
import soccerdata as sd
import structlog

from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)

class TransferMarktScraper(BaseScraper):
    """Extrae valores de mercado de selecciones desde TransferMarkt."""

    def _fetch(self, league: str = 'FIFA World Cup', season: str = '2026') -> pd.DataFrame:
        try:
            # Obtener datos de la liga/copa específica
            tm = sd.Transfermarkt(leagues=league, seasons=season)
            # Esto devuelve info de equipos en el torneo, que incluye valor
            df = tm.read_team_info()
            return df.reset_index()
        except Exception as e:
            logger.error("Error scrapeando TransferMarkt", error=str(e))
            raise
