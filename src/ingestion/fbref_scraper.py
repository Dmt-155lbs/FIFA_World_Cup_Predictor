"""
Scraper de resultados internacionales de selecciones.

Fuente: dataset público ``martj42/international_results`` (CSV plano alojado en
GitHub) que contiene **todos** los partidos internacionales de selecciones desde
1872 (fecha, equipos, marcador, sede neutral y torneo).

¿Por qué no FBref directo?
    Las versiones modernas de ``soccerdata`` acceden a FBref mediante un navegador
    headless (SeleniumBase + undetected-chromedriver) para sortear la protección
    Cloudflare. El contenedor de producción no incluye Chrome, por lo que esa ruta
    falla con "Chrome not found!". Este dataset entrega los mismos datos esenciales
    para FACT_MATCH sin dependencias de navegador y con nombres de selección que
    mapean limpiamente contra ``DIM_TEAM``.

La clase conserva el nombre ``FBrefScraper`` y la interfaz
``fetch_matches_with_xg``/``_fetch`` para no romper el CLI ni el scheduler.
"""
from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd
import requests
import structlog

from src.config import get_settings
from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)


class FBrefScraper(BaseScraper):
    """Obtiene resultados de partidos de selecciones desde un CSV robusto."""

    RESULTS_URL = (
        "https://raw.githubusercontent.com/martj42/"
        "international_results/master/results.csv"
    )

    # Torneos del dataset que mapean a una competición sembrada en DIM_COMPETITION.
    # Los no listados se etiquetan como 'International Friendly' (que sí existe en
    # DIM_COMPETITION), garantizando que el match resuelva su competition_id.
    _COMP_MAP = {
        "FIFA World Cup": "FIFA World Cup",
        "UEFA Nations League": "UEFA Nations League",
        "AFC Asian Cup": "AFC Asian Cup",
    }

    # Torneos de eliminación directa (para el flag is_knockout).
    _KNOCKOUT_TOURNAMENTS = {"FIFA World Cup"}

    # Normalización de nombres del dataset → nombre canónico en DIM_TEAM.
    # Red de seguridad: la mayoría ya coincide, pero cubrimos variantes.
    _TEAM_NORM = {
        "Türkiye": "Turkey",
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
        "USA": "United States",
        "Côte d'Ivoire": "Ivory Coast",
        "Cape Verde Islands": "Cape Verde",
        "China PR": "China",
    }

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()

    def _fetch(
        self,
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Implementación del método abstracto: delega en el descargador."""
        return self.fetch_matches_with_xg(leagues, seasons)

    def fetch_matches_with_xg(
        self,
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Descarga y normaliza los resultados internacionales.

        Returns
        -------
        pd.DataFrame
            Columnas alineadas con ``DataLoader.load_matches``:
            ``home_team, away_team, match_date, home_goals, away_goals,
            is_neutral, venue, competition, is_knockout``.
        """
        start_year = self.settings.data_start_year
        logger.info(
            "Descargando resultados internacionales",
            url=self.RESULTS_URL,
            desde=start_year,
        )

        resp = requests.get(
            self.RESULTS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MundialBot/1.0)"},
            timeout=60,
        )
        resp.raise_for_status()

        raw = pd.read_csv(io.StringIO(resp.text))

        # Solo partidos jugados (con marcador) y dentro de la ventana temporal.
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date", "home_score", "away_score"])
        raw = raw[raw["date"].dt.year >= start_year]

        home = raw["home_team"].astype(str).str.strip().replace(self._TEAM_NORM)
        away = raw["away_team"].astype(str).str.strip().replace(self._TEAM_NORM)

        df = pd.DataFrame(
            {
                "home_team": home,
                "away_team": away,
                "match_date": raw["date"].dt.date,
                "home_goals": raw["home_score"].astype(int),
                "away_goals": raw["away_score"].astype(int),
                "is_neutral": raw["neutral"].astype(bool),
                "venue": raw["city"].astype(str).str.strip(),
                "competition": raw["tournament"].map(
                    lambda t: self._COMP_MAP.get(t, "International Friendly")
                ),
                "is_knockout": raw["tournament"].isin(
                    self._KNOCKOUT_TOURNAMENTS
                ),
            }
        ).reset_index(drop=True)

        logger.info(
            "Resultados internacionales descargados",
            partidos=len(df),
            desde=str(start_year),
        )
        return df
