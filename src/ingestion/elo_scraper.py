"""
Scraper de eloratings.net.
Extrae los ratings Elo históricos de las selecciones nacionales.

El archivo World.tsv de eloratings.net contiene datos tabulados con
columnas separadas por tabuladores. Este módulo descarga, parsea y
normaliza el TSV a un DataFrame estándar para el pipeline.
"""
import io
from datetime import datetime, date

import pandas as pd
import requests
import structlog

from src.ingestion.base_scraper import BaseScraper

logger = structlog.get_logger(__name__)


class EloScraper(BaseScraper):
    """Obtiene ratings Elo desde eloratings.net.

    Descarga el archivo TSV público con los ratings actuales de todas
    las selecciones y lo parsea a un DataFrame normalizado.
    """

    BASE_URL = "https://www.eloratings.net"

    _USER_AGENT = "Mozilla/5.0 (compatible; MundialBot/1.0)"

    # Mapeo de nombres alternativos → nombre canónico FIFA
    _NAME_NORMALIZATION: dict[str, str] = {
        "USA": "United States",
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
        "IR Iran": "Iran",
        "Côte d'Ivoire": "Ivory Coast",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
        "Chinese Taipei": "Taiwan",
    }

    def _fetch(self, date: str | None = None) -> pd.DataFrame:
        """Descarga y parsea el TSV de ratings Elo.

        Parameters
        ----------
        date : str | None
            Fecha en formato 'YYYY-MM-DD' para obtener ratings
            históricos.  Si ``None``, obtiene los ratings actuales.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas:
            ``['team', 'elo_rating', 'elo_delta', 'rating_date']``.
        """
        url = f"{self.BASE_URL}/World.tsv"
        if date:
            url = f"{self.BASE_URL}/{date}/World.tsv"

        logger.info("Descargando ratings Elo", url=url)

        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MundialBot/1.0)"},
            timeout=30,
        )
        response.raise_for_status()

        # Parsear TSV crudo a DataFrame
        raw_text = response.text
        df = self._parse_tsv(raw_text)

        # Agregar fecha de rating
        rating_date = (
            datetime.strptime(date, "%Y-%m-%d").date()
            if date
            else datetime.now().date()
        )
        df["rating_date"] = rating_date

        # Normalizar nombres de equipos
        df["team"] = df["team"].replace(self._NAME_NORMALIZATION)

        logger.info(
            "Ratings Elo descargados",
            equipos=len(df),
            fecha=str(rating_date),
        )

        return df

    def _parse_tsv(self, raw_text: str) -> pd.DataFrame:
        """Parsea el texto TSV crudo al DataFrame estándar.

        Maneja variaciones en el formato del archivo:
        - Con o sin cabecera
        - Columnas extras (ranking, confederación, etc.)
        - Líneas vacías o comentarios

        Parameters
        ----------
        raw_text : str
            Contenido crudo del archivo TSV.

        Returns
        -------
        pd.DataFrame
            DataFrame limpio con columnas
            ``['team', 'elo_rating', 'elo_delta']``.
        """
        # Filtrar líneas vacías y comentarios
        lines = [
            line for line in raw_text.strip().split("\n")
            if line.strip() and not line.startswith("#")
        ]

        if not lines:
            logger.warning("TSV vacío recibido de eloratings.net")
            return pd.DataFrame(
                columns=["team", "elo_rating", "elo_delta"]
            )

        # Intentar parsear con pandas
        try:
            df = pd.read_csv(
                io.StringIO("\n".join(lines)),
                sep="\t",
                header=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.error("Error parseando TSV", error=str(e))
            raise

        # Identificar columnas por contenido
        # Estructura típica: Rank | Team | Elo | Delta | ...
        if len(df.columns) >= 4:
            # Columnas: rank, team, elo, delta, ...
            result = pd.DataFrame({
                "team": df.iloc[:, 1].astype(str).str.strip(),
                "elo_rating": pd.to_numeric(
                    df.iloc[:, 2], errors="coerce"
                ),
                "elo_delta": pd.to_numeric(
                    df.iloc[:, 3], errors="coerce"
                ),
            })
        elif len(df.columns) == 3:
            result = pd.DataFrame({
                "team": df.iloc[:, 0].astype(str).str.strip(),
                "elo_rating": pd.to_numeric(
                    df.iloc[:, 1], errors="coerce"
                ),
                "elo_delta": pd.to_numeric(
                    df.iloc[:, 2], errors="coerce"
                ),
            })
        elif len(df.columns) == 2:
            result = pd.DataFrame({
                "team": df.iloc[:, 0].astype(str).str.strip(),
                "elo_rating": pd.to_numeric(
                    df.iloc[:, 1], errors="coerce"
                ),
                "elo_delta": 0.0,
            })
        else:
            raise ValueError(
                f"Formato TSV inesperado: {len(df.columns)} columnas"
            )

        # Limpiar filas con datos no numéricos
        result = result.dropna(subset=["elo_rating"])
        result["elo_delta"] = result["elo_delta"].fillna(0.0)

        # Filtrar filas donde 'team' parece un encabezado
        result = result[
            ~result["team"].str.lower().isin(
                ["team", "country", "rank", ""]
            )
        ]

        return result.reset_index(drop=True)
