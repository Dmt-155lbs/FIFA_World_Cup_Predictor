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

    # Fecha de valoración del snapshot Elo. CLAVE: eloratings.net sólo expone el
    # Elo ACTUAL (un snapshot), pero la vista vw_feature_store aplica el Elo con
    # ``rating_date <= match_date``. Si se fechara HOY, ningún partido histórico
    # (2014–2026) heredaría Elo y la feature quedaría muerta (≈0 en todas las
    # filas), dejando al modelo sin su mejor señal de fuerza → optimismo ciego
    # con los underdogs. Igual que SoFIFA, se fija TEMPRANO para que el Elo
    # actual sirva de proxy estático de fuerza en TODOS los partidos.
    SNAPSHOT_DATE = date(2014, 1, 1)

    # ------------------------------------------------------------------ #
    # Mapeo de los códigos de 2 letras de eloratings.net → nombre canónico
    # FIFA usado en DIM_TEAM/seed_data. eloratings expone World.tsv con la
    # selección identificada por un código de 2 letras (mayormente ISO
    # 3166-1 alpha-2, con excepciones para las "home nations" británicas:
    # EN=England, WA=Wales, SC=Scotland, NI=Northern Ireland).
    # Sin este mapeo el código (p. ej. "ES") nunca casa con DIM_TEAM y la
    # tabla FACT_ELO_HISTORY queda vacía.
    # Se cubren los 48 del Mundial + algunas selecciones frecuentes.
    # ------------------------------------------------------------------ #
    _ELO_CODE_TO_NAME: dict[str, str] = {
        # CONMEBOL
        "AR": "Argentina", "BR": "Brazil", "UY": "Uruguay",
        "CO": "Colombia", "EC": "Ecuador", "PY": "Paraguay",
        "CL": "Chile", "PE": "Peru", "VE": "Venezuela", "BO": "Bolivia",
        # UEFA
        "FR": "France", "ES": "Spain", "EN": "England", "PT": "Portugal",
        "NL": "Netherlands", "BE": "Belgium", "DE": "Germany", "IT": "Italy",
        "HR": "Croatia", "DK": "Denmark", "CH": "Switzerland", "AT": "Austria",
        "RS": "Serbia", "WA": "Wales", "TR": "Turkey", "UA": "Ukraine",
        "PL": "Poland", "SE": "Sweden", "NO": "Norway", "CZ": "Czech Republic",
        "SC": "Scotland", "BA": "Bosnia and Herzegovina",
        "HU": "Hungary", "RO": "Romania", "GR": "Greece",
        # CONCACAF
        "US": "United States", "MX": "Mexico", "CA": "Canada",
        "JM": "Jamaica", "PA": "Panama", "HN": "Honduras",
        "CR": "Costa Rica", "TT": "Trinidad and Tobago",
        "HT": "Haiti", "CW": "Curacao",
        # AFC
        "JP": "Japan", "KR": "South Korea", "AU": "Australia", "IR": "Iran",
        "SA": "Saudi Arabia", "QA": "Qatar", "IQ": "Iraq", "UZ": "Uzbekistan",
        "ID": "Indonesia", "CN": "China", "AE": "United Arab Emirates",
        "JO": "Jordan",
        # CAF
        "MA": "Morocco", "SN": "Senegal", "NG": "Nigeria", "EG": "Egypt",
        "CM": "Cameroon", "ZA": "South Africa", "DZ": "Algeria", "ML": "Mali",
        "CI": "Ivory Coast", "TN": "Tunisia", "GH": "Ghana",
        "CV": "Cape Verde", "CD": "DR Congo",
        # OFC
        "NZ": "New Zealand",
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

        # Agregar fecha de rating. Sin fecha explícita usamos SNAPSHOT_DATE
        # (temprana) para que el Elo actual aplique como proxy estático a todos
        # los partidos históricos vía la vista (rating_date <= match_date).
        rating_date = (
            datetime.strptime(date, "%Y-%m-%d").date()
            if date
            else self.SNAPSHOT_DATE
        )
        df["rating_date"] = rating_date

        # Traducir el código de 2 letras de eloratings → nombre canónico FIFA.
        # Las selecciones fuera del mapeo (no relevantes para el Mundial) se
        # descartan: su código no casaría con DIM_TEAM de todos modos.
        df["team"] = df["team"].str.strip().str.upper().map(self._ELO_CODE_TO_NAME)
        antes = len(df)
        df = df.dropna(subset=["team"]).reset_index(drop=True)

        logger.info(
            "Ratings Elo descargados",
            equipos_mapeados=len(df),
            equipos_totales=antes,
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

        # Identificar columnas por contenido.
        # Estructura REAL de World.tsv (verificada): la fila es
        #   rank | rank2 | código_2_letras | elo | <muchas columnas más>
        # es decir el código de equipo está en el índice 2 y el Elo en el 3.
        # (El parser anterior asumía team=idx1/elo=idx2, lo que tomaba un
        #  número como nombre y el código como Elo → todo se descartaba.)
        if len(df.columns) >= 4:
            result = pd.DataFrame({
                "team": df.iloc[:, 2].astype(str).str.strip(),
                "elo_rating": pd.to_numeric(
                    df.iloc[:, 3], errors="coerce"
                ),
                # No hay una columna de delta limpia y estable en el TSV;
                # no es crítica para el feature store, se deja en 0.
                "elo_delta": 0.0,
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
