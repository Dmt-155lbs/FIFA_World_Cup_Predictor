"""
Scraper de ratings FIFA de selecciones nacionales (estilo SoFIFA) SIN navegador.

¿Por qué no SoFIFA directo / soccerdata?
    ``soccerdata.SoFIFA`` accede a sofifa.com mediante un navegador headless para
    sortear Cloudflare; el contenedor de producción no incluye Chrome
    ("Chrome not found!"). Y un ``requests.get`` directo a sofifa.com devuelve
    HTTP 403 (anti-bot). Igual que con FBref, resolvemos esto con una fuente
    plana y pública alcanzable por ``requests``:

    Usamos el *FIFA complete player dataset* (scrapeado originalmente DE sofifa.com,
    misma procedencia) alojado como CSV en GitHub, y derivamos el rating de cada
    SELECCIÓN agregando a sus jugadores por posición:

        overall  = media de los mejores jugadores del país
        attack   = media de los mejores delanteros
        midfield = media de los mejores mediocampistas
        defence  = media de los mejores defensas (incl. portero)

    El resultado es un rating de selección comparable al que SoFIFA publica en
    sus páginas de equipos nacionales, pero calculado por nosotros sin navegador.

La clase conserva el nombre ``SoFIFAScraper`` y la interfaz ``_fetch`` para no
romper el CLI ni el scheduler.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests
import structlog

from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)


class SoFIFAScraper(BaseScraper):
    """Deriva ratings FIFA de selecciones desde un CSV público de jugadores."""

    # FIFA complete player dataset (origen sofifa.com), CSV plano en GitHub.
    # Columnas relevantes: ``Nationality``, ``Overall``, ``Position``.
    RATINGS_URL = (
        "https://raw.githubusercontent.com/densaiko/"
        "data_science_learning/main/dataset/fifa_dataset.csv"
    )

    # Fecha de valoración del snapshot. Se fija TEMPRANO (inicio de la ventana
    # de datos) porque la vista vw_feature_store toma el rating con
    # ``valuation_date <= match_date``; así TODOS los partidos históricos
    # (2014–2026) heredan el rating como un proxy estático de calidad de plantel.
    SNAPSHOT_DATE = date(2014, 1, 1)

    # Tamaños de muestra para la agregación por selección.
    _SQUAD_SIZE = 23        # jugadores considerados para el overall
    _GROUP_SIZE = 5         # mejores por línea (att/mid/def)

    # Mapeo: nombre de país en el dataset → nombre canónico en DIM_TEAM/seed_data.
    # 46 de los 48 coinciden tal cual; sólo estos dos difieren.
    _NATIONALITY_NORM: dict[str, str] = {
        "Korea Republic": "South Korea",
        "Bosnia Herzegovina": "Bosnia and Herzegovina",
        # Redes de seguridad para variantes comunes de otras selecciones:
        "Korea DPR": "Korea DPR",
        "China PR": "China",
    }

    # Clasificación de las posiciones del dataset en líneas.
    _ATTACK = {"ST", "CF", "LW", "RW", "LF", "RF", "LS", "RS"}
    _MIDFIELD = {
        "CAM", "CM", "CDM", "LAM", "RAM", "LCM", "RCM",
        "LDM", "RDM", "LM", "RM",
    }
    _DEFENCE = {"CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB", "GK"}

    def _fetch(self, **kwargs) -> pd.DataFrame:
        """Descarga el CSV de jugadores y agrega los ratings por selección.

        Returns
        -------
        pd.DataFrame
            Columnas alineadas con ``DataLoader.load_fifa_ratings``:
            ``['team', 'overall_rating', 'attack_rating', 'midfield_rating',
               'defence_rating', 'valuation_date']``.
        """
        logger.info("Descargando dataset FIFA de jugadores", url=self.RATINGS_URL)

        resp = requests.get(
            self.RATINGS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MundialBot/1.0)"},
            timeout=60,
        )
        resp.raise_for_status()

        raw = pd.read_csv(io.StringIO(resp.text))

        # Normalizar columnas mínimas necesarias.
        needed = {"Nationality", "Overall", "Position"}
        if not needed.issubset(raw.columns):
            raise ValueError(
                f"El dataset FIFA no tiene las columnas esperadas {needed}; "
                f"recibidas: {list(raw.columns)[:15]}"
            )

        df = raw[["Nationality", "Overall", "Position"]].copy()
        df = df.dropna(subset=["Nationality", "Overall", "Position"])
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce")
        df = df.dropna(subset=["Overall"])

        # Traducir el país al nombre canónico (los no mapeados se dejan igual:
        # la mayoría ya coincide con DIM_TEAM; los que no, el loader los omite).
        df["team"] = df["Nationality"].astype(str).str.strip().map(
            lambda n: self._NATIONALITY_NORM.get(n, n)
        )
        df["Position"] = df["Position"].astype(str).str.strip().str.upper()

        rows: list[dict] = []
        for team, grp in df.groupby("team"):
            grp = grp.sort_values("Overall", ascending=False)

            overall = self._mean_top(grp["Overall"], self._SQUAD_SIZE)
            attack = self._line_rating(grp, self._ATTACK, overall)
            midfield = self._line_rating(grp, self._MIDFIELD, overall)
            defence = self._line_rating(grp, self._DEFENCE, overall)

            rows.append({
                "team": team,
                "overall_rating": overall,
                "attack_rating": attack,
                "midfield_rating": midfield,
                "defence_rating": defence,
                "valuation_date": self.SNAPSHOT_DATE,
            })

        result = pd.DataFrame(rows)
        logger.info(
            "Ratings FIFA de selecciones derivados",
            selecciones=len(result),
            fecha=str(self.SNAPSHOT_DATE),
        )
        return result

    @staticmethod
    def _mean_top(series: pd.Series, n: int) -> float:
        """Media (redondeada) de los ``n`` valores más altos de la serie."""
        top = series.head(n)
        return round(float(top.mean()), 1) if len(top) else 0.0

    def _line_rating(
        self, grp: pd.DataFrame, positions: set[str], fallback: float
    ) -> float:
        """Rating de una línea = media de los mejores jugadores de esas posiciones.

        Si la selección no tiene jugadores en esa línea dentro del dataset, se
        usa el overall como respaldo para no dejar la feature en cero.
        """
        line = grp[grp["Position"].isin(positions)]
        if line.empty:
            return fallback
        return self._mean_top(line["Overall"], self._GROUP_SIZE)
