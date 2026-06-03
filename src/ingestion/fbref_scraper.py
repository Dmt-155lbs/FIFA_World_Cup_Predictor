"""
Scraper de FBref usando la librería soccerdata.
Extrae resultados históricos y métricas avanzadas (xG).
"""
import pandas as pd
import soccerdata as sd
import structlog
from typing import List

from src.ingestion.base_scraper import BaseScraper
from src.config import get_settings

logger = structlog.get_logger(__name__)

class FBrefScraper(BaseScraper):
    """
    Obtiene datos de partidos (resultados y xG) desde FBref.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        # Mapeo de competiciones internacionales comunes en FBref
        self.leagues = ['FIFA World Cup', 'Copa America', 'UEFA Euro', 
                        'UEFA Nations League', 'Africa Cup of Nations']

    def _fetch(self, leagues: List[str] = None, seasons: List[str] = None) -> pd.DataFrame:
        """
        Método base abstracto implementado. 
        En este caso usamos fbref, delegando a fetch_matches_with_xg.
        """
        return self.fetch_matches_with_xg(leagues, seasons)

    def fetch_matches_with_xg(self, leagues: List[str] = None, seasons: List[str] = None) -> pd.DataFrame:
        """
        Obtiene el calendario y xG combinados.
        """
        leagues = leagues or self.leagues
        seasons = seasons or [f"{year}" for year in range(self.settings.data_start_year, 2027)]
        
        try:
            fbref = sd.FBref(leagues=leagues, seasons=seasons)
            # schedule_df contiene resultados básicos e info del partido
            schedule_df = fbref.read_schedule()
            
            # Para partidos internacionales post 2018 (generalmente), FBref tiene xG
            # Hacemos reset_index porque soccerdata devuelve MultiIndex
            df = schedule_df.reset_index()
            
            # Renombrar para alinear con Pydantic validators (se hará en otro paso)
            return df
        except Exception as e:
            logger.error("Error scrapeando FBref", error=str(e))
            raise
