"""
Scrapers de cuotas de apuestas en vivo vía The Odds API.
"""
import pandas as pd
import requests
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.ingestion.base_scraper import BaseScraper
from src.config import get_settings

logger = structlog.get_logger(__name__)

# The Odds API devuelve nombres en inglés común. Los normalizamos a los
# nombres CANÓNICOS de DIM_TEAM (ver src/ingestion/seed_data.py). Sólo se
# remapean las variantes que difieren; el resto coincide tal cual y
# resolve_team_id las resuelve (exacta o case-insensitive). Mapear a una
# variante inexistente (p. ej. "Korea Republic", "IR Iran") haría que el
# fixture se descartara silenciosamente al no encontrar el team_id.
TEAM_NAME_MAPPING = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Curaçao": "Curacao",
}

def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAPPING.get(name, name)

class OddsScraperLive(BaseScraper):
    """
    Consulta The Odds API para cuotas pre-partido.
    Calcula un consenso de las casas de apuestas europeas.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.api_key = self.settings.odds_api_key
        self.base_url = self.settings.odds_api_base_url

    def _fetch(self, sport: str = 'soccer_fifa_world_cup') -> pd.DataFrame:
        if not self.api_key:
            logger.error("ODDS_API_KEY no configurada en las variables de entorno.")
            return pd.DataFrame()
            
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'eu', # Solo europa para ahorrar créditos,
            'markets': 'h2h,totals', # Victoria/Empate y Goles +/- 2.5
            'oddsFormat': 'decimal'
        }
        
        logger.info(f"Consultando The Odds API para {sport}")
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_consensus(data)
        except Exception as e:
            logger.error("Error consultando The Odds API", error=str(e))
            raise

    def _parse_consensus(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for match in data:
            home_team = normalize_team_name(match.get('home_team', ''))
            away_team = normalize_team_name(match.get('away_team', ''))
            commence_time = pd.to_datetime(match.get('commence_time')).tz_localize(None)
            
            # Recolectar cuotas
            h2h_home, h2h_draw, h2h_away = [], [], []
            over_25, under_25 = [], []
            
            for bookie in match.get('bookmakers', []):
                for market in bookie.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == match.get('home_team'):
                                h2h_home.append(outcome['price'])
                            elif outcome.get('name') == match.get('away_team'):
                                h2h_away.append(outcome['price'])
                            elif outcome.get('name') == 'Draw':
                                h2h_draw.append(outcome['price'])
                                
                    elif market['key'] == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('point') == 2.5:
                                if outcome.get('name') == 'Over':
                                    over_25.append(outcome['price'])
                                elif outcome.get('name') == 'Under':
                                    under_25.append(outcome['price'])
            
            if not h2h_home or not h2h_away:
                continue
                
            # Calcular consenso (promedio)
            avg_home = sum(h2h_home) / len(h2h_home)
            avg_draw = sum(h2h_draw) / len(h2h_draw) if h2h_draw else 0.0
            avg_away = sum(h2h_away) / len(h2h_away)
            avg_over = sum(over_25) / len(over_25) if over_25 else 0.0
            
            rows.append({
                "home_team": home_team,
                "away_team": away_team,
                "date": commence_time,
                "bookmaker": "Consensus",
                "odds_home": round(avg_home, 2),
                "odds_draw": round(avg_draw, 2),
                "odds_away": round(avg_away, 2),
                "odds_over25": round(avg_over, 2),
                "last_update": datetime.utcnow()
            })
            
        return pd.DataFrame(rows)
