"""
Scrapers de cuotas de apuestas (Históricas y en Vivo).
"""
import pandas as pd
import requests
import structlog
from typing import Optional

from src.ingestion.base_scraper import BaseScraper
from src.config import get_settings

logger = structlog.get_logger(__name__)

class OddsScraperHistorical(BaseScraper):
    """
    Descarga CSVs históricos desde football-data.co.uk para backtesting.
    """
    
    def _fetch(self, league: str = 'W', season: str = '2223') -> pd.DataFrame:
        # 'W' usualmente no está directo, pero suponemos mapeos (ej: ligas top). 
        # Para el mundial, a veces proveen un csv específico.
        # Aquí definimos la lógica de lectura genérica:
        base_url = "https://www.football-data.co.uk/mmz4281"
        url = f"{base_url}/{season}/{league}.csv"
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            logger.error(f"Error descargando odds de {url}", error=str(e))
            raise


class OddsScraperLive(BaseScraper):
    """
    Consulta The Odds API para cuotas en vivo durante el torneo.
    Limitado para proteger la cuota gratuita.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.api_key = self.settings.odds_api_key
        self.base_url = self.settings.odds_api_base_url

    def _fetch(self, sport: str = 'soccer_fifa_world_cup') -> pd.DataFrame:
        if not self.api_key:
            logger.error("ODDS_API_KEY no configurada.")
            return pd.DataFrame()
            
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'eu,uk', # Casas de apuesta principales
            'markets': 'h2h', # Home, Draw, Away
            'oddsFormat': 'decimal'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            # Convertir JSON complejo a un DataFrame manejable
            return pd.json_normalize(data)
        except Exception as e:
            logger.error("Error consultando The Odds API", error=str(e))
            raise
