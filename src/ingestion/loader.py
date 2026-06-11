"""
DataLoader — El "puente" ETL que conecta scrapers con la base de datos.

Este módulo implementa la capa LOAD del pipeline ETL:
  Scrapers (DataFrame) → Validators (Pydantic) → DataLoader → SQL Server

Cada método público toma un DataFrame crudo del scraper, lo valida fila por
fila usando los modelos Pydantic de ``validators.py``, resuelve las claves
foráneas (team_id, competition_id, match_id) mediante lookups en las tablas
DIM_*, y ejecuta INSERT/UPSERT idempotentes en las tablas FACT_*.

Autor: Mundial 2026 Team
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ingestion.validators import (
    EloHistoryValidator,
    FifaRatingValidator,
    MatchValidator,
    MatchXGValidator,
    OddsValidator,
    TeamValidator,
)
from src.utils.db import get_engine, get_session

log = structlog.get_logger(__name__)


class DataLoader:
    """Carga datos validados desde DataFrames a SQL Server.

    Implementa lookups de dimensiones, validación Pydantic y upserts
    idempotentes para cada tabla FACT del esquema estrella.

    Parameters
    ----------
    engine : Engine | None
        Motor SQLAlchemy.  Si ``None``, se usa el singleton global.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine: Engine = engine or get_engine()
        # Cachés de lookups para evitar queries repetidas
        self._team_cache: dict[str, int] = {}
        self._competition_cache: dict[str, int] = {}
        log.info("DataLoader inicializado")

    # ================================================================== #
    #  SEED — Poblado de dimensiones                                       #
    # ================================================================== #

    def seed_teams(
        self,
        teams: list[tuple[str, str, str, int]],
    ) -> int:
        """Inserta equipos en DIM_TEAM si no existen (por fifa_code).

        Parameters
        ----------
        teams : list[tuple]
            Lista de ``(team_name, fifa_code, confederation, fifa_ranking)``.

        Returns
        -------
        int
            Número de equipos insertados (nuevos).
        """
        inserted = 0
        with get_session() as session:
            for team_name, fifa_code, confederation, ranking in teams:
                # Validar con Pydantic
                validated = TeamValidator(
                    team_name=team_name,
                    fifa_code=fifa_code,
                    confederation=confederation,
                    fifa_ranking=ranking,
                )

                # Idempotencia: solo insertar si no existe
                existing = session.execute(
                    text(
                        "SELECT [team_id] FROM [mundial].[DIM_TEAM] "
                        "WHERE [fifa_code] = :code"
                    ),
                    {"code": validated.fifa_code},
                ).scalar()

                if existing is None:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[DIM_TEAM]
                            ([team_name], [fifa_code], [confederation],
                             [fifa_ranking])
                            VALUES (:name, :code, :conf, :rank)
                            """
                        ),
                        {
                            "name": validated.team_name,
                            "code": validated.fifa_code,
                            "conf": validated.confederation,
                            "rank": validated.fifa_ranking,
                        },
                    )
                    inserted += 1

        # Invalidar caché
        self._team_cache.clear()

        log.info(
            "Seed de equipos completado",
            insertados=inserted,
            total=len(teams),
        )
        return inserted

    def seed_competitions(
        self,
        competitions: list[tuple[str, str, str]],
    ) -> int:
        """Inserta competiciones en DIM_COMPETITION si no existen.

        Parameters
        ----------
        competitions : list[tuple]
            Lista de ``(competition_name, season, stage)``.

        Returns
        -------
        int
            Número de competiciones insertadas (nuevas).
        """
        inserted = 0
        with get_session() as session:
            for comp_name, season, stage in competitions:
                existing = session.execute(
                    text(
                        "SELECT [competition_id] "
                        "FROM [mundial].[DIM_COMPETITION] "
                        "WHERE [competition_name] = :name "
                        "AND [season] = :season"
                    ),
                    {"name": comp_name, "season": season},
                ).scalar()

                if existing is None:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[DIM_COMPETITION]
                            ([competition_name], [season], [stage])
                            VALUES (:name, :season, :stage)
                            """
                        ),
                        {
                            "name": comp_name,
                            "season": season,
                            "stage": stage,
                        },
                    )
                    inserted += 1

        # Invalidar caché
        self._competition_cache.clear()

        log.info(
            "Seed de competiciones completado",
            insertadas=inserted,
            total=len(competitions),
        )
        return inserted

    # ================================================================== #
    #  LOOKUPS — Resolución de claves foráneas                             #
    # ================================================================== #

    def resolve_team_id(self, team_name: str) -> int | None:
        """Resuelve un nombre de equipo a su team_id en DIM_TEAM.

        Busca primero en caché, luego por ``team_name`` exacto, y
        finalmente por ``team_name`` con LIKE para coincidencias parciales.

        Parameters
        ----------
        team_name : str
            Nombre del equipo a resolver.

        Returns
        -------
        int | None
            ``team_id`` o ``None`` si no se encuentra.
        """
        if not team_name or team_name == "UNKNOWN":
            return None

        # Caché en memoria
        if team_name in self._team_cache:
            return self._team_cache[team_name]

        with self._engine.connect() as conn:
            # Búsqueda exacta
            result = conn.execute(
                text(
                    "SELECT [team_id] FROM [mundial].[DIM_TEAM] "
                    "WHERE [team_name] = :name"
                ),
                {"name": team_name},
            ).scalar()

            if result is None:
                # Búsqueda parcial (insensible a mayúsculas)
                result = conn.execute(
                    text(
                        "SELECT TOP 1 [team_id] "
                        "FROM [mundial].[DIM_TEAM] "
                        "WHERE LOWER([team_name]) = LOWER(:name)"
                    ),
                    {"name": team_name},
                ).scalar()

        if result is not None:
            self._team_cache[team_name] = result

        return result

    def resolve_competition_id(
        self, competition_name: str, season: str = "UNKNOWN"
    ) -> int | None:
        """Resuelve una competición a su competition_id.

        Parameters
        ----------
        competition_name : str
            Nombre de la competición.
        season : str
            Temporada o edición.

        Returns
        -------
        int | None
            ``competition_id`` o ``None``.
        """
        cache_key = f"{competition_name}|{season}"
        if cache_key in self._competition_cache:
            return self._competition_cache[cache_key]

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT [competition_id] "
                    "FROM [mundial].[DIM_COMPETITION] "
                    "WHERE [competition_name] = :name "
                    "AND [season] = :season"
                ),
                {"name": competition_name, "season": season},
            ).scalar()

            # Fallback: buscar solo por nombre
            if result is None:
                result = conn.execute(
                    text(
                        "SELECT TOP 1 [competition_id] "
                        "FROM [mundial].[DIM_COMPETITION] "
                        "WHERE [competition_name] = :name "
                        "ORDER BY [competition_id] DESC"
                    ),
                    {"name": competition_name},
                ).scalar()

        if result is not None:
            self._competition_cache[cache_key] = result

        return result

    def resolve_match_id(
        self,
        match_date: date,
        home_team_id: int,
        away_team_id: int,
    ) -> int | None:
        """Resuelve un partido a su match_id por clave natural.

        El índice único ``UQ_MATCH_IDEMPOTENT`` garantiza unicidad de
        ``(match_date, home_team_id, away_team_id)``.

        Returns
        -------
        int | None
            ``match_id`` o ``None``.
        """
        with self._engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT [match_id] FROM [mundial].[FACT_MATCH] "
                    "WHERE [match_date] = :dt "
                    "AND [home_team_id] = :home "
                    "AND [away_team_id] = :away"
                ),
                {"dt": match_date, "home": home_team_id, "away": away_team_id},
            ).scalar()

    # ================================================================== #
    #  LOAD — Carga de tablas FACT                                         #
    # ================================================================== #

    def load_matches(
        self,
        df: pd.DataFrame,
        competition_name: str = "International Friendly",
        season: str = "2014-2026",
    ) -> dict[str, int]:
        """Carga partidos desde un DataFrame a FACT_MATCH.

        El DataFrame debe contener columnas que mapeen a los campos
        de ``MatchValidator``. Los nombres de equipo se resuelven
        automáticamente a ``team_id`` via lookup en ``DIM_TEAM``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas: ``home_team``, ``away_team``,
            ``match_date``, ``home_goals``, ``away_goals`` (mínimo).
            Opcionalmente: ``venue``, ``is_neutral``, ``is_knockout``,
            ``attendance``, ``competition``.
        competition_name : str
            Nombre por defecto de la competición si no está en el DF.
        season : str
            Temporada por defecto.

        Returns
        -------
        dict[str, int]
            ``{"inserted": N, "skipped": M, "errors": E}``.
        """
        inserted = skipped = errors = 0

        for idx, row in df.iterrows():
            try:
                # Resolver claves foráneas
                home_name = str(row.get("home_team", ""))
                away_name = str(row.get("away_team", ""))
                home_id = self.resolve_team_id(home_name)
                away_id = self.resolve_team_id(away_name)

                if home_id is None or away_id is None:
                    log.debug(
                        "Equipo no encontrado, omitiendo",
                        home=home_name,
                        away=away_name,
                    )
                    skipped += 1
                    continue

                # Resolver competición
                comp_name = str(
                    row.get("competition", competition_name)
                )
                comp_season = str(row.get("season", season))
                comp_id = self.resolve_competition_id(
                    comp_name, comp_season
                )
                if comp_id is None:
                    comp_id = self.resolve_competition_id(comp_name)
                if comp_id is None:
                    log.debug(
                        "Competición no encontrada",
                        competicion=comp_name,
                    )
                    skipped += 1
                    continue

                # Validar con Pydantic
                match = MatchValidator(
                    competition_id=comp_id,
                    match_date=row.get("match_date"),
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_goals=row.get("home_goals", 0),
                    away_goals=row.get("away_goals", 0),
                    venue=row.get("venue", "UNKNOWN"),
                    is_neutral=bool(row.get("is_neutral", False)),
                    is_knockout=bool(row.get("is_knockout", False)),
                    attendance=row.get("attendance", -1),
                )

                # Idempotencia: verificar si ya existe
                existing = self.resolve_match_id(
                    match.match_date, match.home_team_id, match.away_team_id
                )
                if existing is not None:
                    skipped += 1
                    continue

                # Insertar
                data = match.to_db_dict()
                with get_session() as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[FACT_MATCH]
                            ([competition_id], [match_date], [home_team_id],
                             [away_team_id], [home_goals], [away_goals],
                             [venue], [is_neutral], [is_knockout],
                             [attendance])
                            VALUES (:competition_id, :match_date,
                                    :home_team_id, :away_team_id,
                                    :home_goals, :away_goals, :venue,
                                    :is_neutral, :is_knockout, :attendance)
                            """
                        ),
                        data,
                    )
                inserted += 1

            except Exception as e:
                log.warning(
                    "Error cargando partido",
                    fila=idx,
                    error=str(e),
                )
                errors += 1

        result = {"inserted": inserted, "skipped": skipped, "errors": errors}
        log.info("Carga de partidos completada", **result)
        return result

    def load_elo_history(self, df: pd.DataFrame) -> dict[str, int]:
        """Carga historial de Elo desde un DataFrame a FACT_ELO_HISTORY.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas: ``team``, ``elo_rating``,
            ``elo_delta``, ``rating_date``.

        Returns
        -------
        dict[str, int]
            ``{"inserted": N, "skipped": M, "errors": E}``.
        """
        inserted = skipped = errors = 0

        for idx, row in df.iterrows():
            try:
                team_name = str(row.get("team", ""))
                team_id = self.resolve_team_id(team_name)

                if team_id is None:
                    skipped += 1
                    continue

                validated = EloHistoryValidator(
                    team_id=team_id,
                    rating_date=row.get("rating_date"),
                    elo_rating=row.get("elo_rating", 0.0),
                    elo_delta=row.get("elo_delta", 0.0),
                )

                # Idempotencia: verificar duplicado por equipo + fecha
                with self._engine.connect() as conn:
                    existing = conn.execute(
                        text(
                            "SELECT [elo_id] "
                            "FROM [mundial].[FACT_ELO_HISTORY] "
                            "WHERE [team_id] = :tid "
                            "AND [rating_date] = :dt"
                        ),
                        {"tid": team_id, "dt": validated.rating_date},
                    ).scalar()

                if existing is not None:
                    skipped += 1
                    continue

                with get_session() as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[FACT_ELO_HISTORY]
                            ([team_id], [rating_date], [elo_rating],
                             [elo_delta])
                            VALUES (:team_id, :rating_date, :elo_rating,
                                    :elo_delta)
                            """
                        ),
                        validated.to_db_dict(),
                    )
                inserted += 1

            except Exception as e:
                log.warning(
                    "Error cargando Elo",
                    fila=idx,
                    error=str(e),
                )
                errors += 1

        result = {"inserted": inserted, "skipped": skipped, "errors": errors}
        log.info("Carga de Elo history completada", **result)
        return result

    def load_fifa_ratings(self, df: pd.DataFrame) -> dict[str, int]:
        """Carga ratings FIFA/SoFIFA a FACT_FIFA_RATING.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas: ``team``, ``overall_rating``,
            ``attack_rating``, ``midfield_rating``, ``defence_rating``,
            ``rating_date`` (o ``valuation_date``).

        Returns
        -------
        dict[str, int]
            ``{"inserted": N, "skipped": M, "errors": E}``.
        """
        inserted = skipped = errors = 0

        for idx, row in df.iterrows():
            try:
                team_name = str(row.get("team", ""))
                team_id = self.resolve_team_id(team_name)

                if team_id is None:
                    skipped += 1
                    continue

                # Aceptar ambos nombres de columna de fecha
                rating_date = row.get(
                    "valuation_date", row.get("rating_date")
                )

                validated = FifaRatingValidator(
                    team_id=team_id,
                    rating_date=rating_date,
                    overall_rating=row.get("overall_rating", 0.0),
                    attack_rating=row.get("attack_rating", 0.0),
                    midfield_rating=row.get("midfield_rating", 0.0),
                    defence_rating=row.get("defence_rating", 0.0),
                )

                # Idempotencia
                with self._engine.connect() as conn:
                    existing = conn.execute(
                        text(
                            "SELECT [rating_id] "
                            "FROM [mundial].[FACT_FIFA_RATING] "
                            "WHERE [team_id] = :tid "
                            "AND [valuation_date] = :dt"
                        ),
                        {"tid": team_id, "dt": validated.rating_date},
                    ).scalar()

                if existing is not None:
                    skipped += 1
                    continue

                data = validated.to_db_dict()
                with get_session() as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[FACT_FIFA_RATING]
                            ([team_id], [valuation_date], [overall_rating],
                             [attack_rating], [midfield_rating],
                             [defence_rating])
                            VALUES (:team_id, :rating_date,
                                    :overall_rating, :attack_rating,
                                    :midfield_rating, :defence_rating)
                            """
                        ),
                        data,
                    )
                inserted += 1

            except Exception as e:
                log.warning(
                    "Error cargando FIFA rating",
                    fila=idx,
                    error=str(e),
                )
                errors += 1

        result = {"inserted": inserted, "skipped": skipped, "errors": errors}
        log.info("Carga de FIFA ratings completada", **result)
        return result

    def load_match_xg(self, df: pd.DataFrame) -> dict[str, int]:
        """Carga datos de xG a FACT_MATCH_XG.

        El DataFrame debe tener columnas que permitan identificar el
        partido (``home_team``, ``away_team``, ``match_date``) y las
        métricas de xG.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con: ``home_team``, ``away_team``, ``match_date``,
            ``home_xg``, ``away_xg``.  Opcionalmente: ``home_xga``,
            ``away_xga``, ``home_npxg``, ``away_npxg``.

        Returns
        -------
        dict[str, int]
            ``{"inserted": N, "skipped": M, "errors": E}``.
        """
        inserted = skipped = errors = 0

        for idx, row in df.iterrows():
            try:
                # Resolver match_id
                home_id = self.resolve_team_id(
                    str(row.get("home_team", ""))
                )
                away_id = self.resolve_team_id(
                    str(row.get("away_team", ""))
                )

                if home_id is None or away_id is None:
                    skipped += 1
                    continue

                match_id = self.resolve_match_id(
                    row.get("match_date"), home_id, away_id
                )

                if match_id is None:
                    skipped += 1
                    continue

                validated = MatchXGValidator(
                    match_id=match_id,
                    home_xg=row.get("home_xg", 0.0),
                    away_xg=row.get("away_xg", 0.0),
                    home_xga=row.get("home_xga", 0.0),
                    away_xga=row.get("away_xga", 0.0),
                    home_npxg=row.get("home_npxg", 0.0),
                    away_npxg=row.get("away_npxg", 0.0),
                )

                # Idempotencia: un xG por partido
                with self._engine.connect() as conn:
                    existing = conn.execute(
                        text(
                            "SELECT [xg_id] "
                            "FROM [mundial].[FACT_MATCH_XG] "
                            "WHERE [match_id] = :mid"
                        ),
                        {"mid": match_id},
                    ).scalar()

                if existing is not None:
                    skipped += 1
                    continue

                with get_session() as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[FACT_MATCH_XG]
                            ([match_id], [home_xg], [away_xg],
                             [home_xga], [away_xga],
                             [home_npxg], [away_npxg])
                            VALUES (:match_id, :home_xg, :away_xg,
                                    :home_xga, :away_xga,
                                    :home_npxg, :away_npxg)
                            """
                        ),
                        validated.to_db_dict(),
                    )
                inserted += 1

            except Exception as e:
                log.warning("Error cargando xG", fila=idx, error=str(e))
                errors += 1

        result = {"inserted": inserted, "skipped": skipped, "errors": errors}
        log.info("Carga de xG completada", **result)
        return result

    def load_odds(self, df: pd.DataFrame) -> dict[str, int]:
        """Carga cuotas de apuestas a FACT_ODDS.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con: ``home_team``, ``away_team``, ``match_date``,
            ``bookmaker``, ``odds_home``, ``odds_draw``, ``odds_away``.
            Opcionalmente: ``odds_over25``.

        Returns
        -------
        dict[str, int]
            ``{"inserted": N, "skipped": M, "errors": E}``.
        """
        inserted = skipped = errors = 0

        for idx, row in df.iterrows():
            try:
                home_id = self.resolve_team_id(
                    str(row.get("home_team", ""))
                )
                away_id = self.resolve_team_id(
                    str(row.get("away_team", ""))
                )

                if home_id is None or away_id is None:
                    skipped += 1
                    continue

                match_id = self.resolve_match_id(
                    row.get("match_date"), home_id, away_id
                )

                if match_id is None:
                    skipped += 1
                    continue

                bookmaker = str(row.get("bookmaker", "UNKNOWN"))

                validated = OddsValidator(
                    match_id=match_id,
                    bookmaker=bookmaker,
                    odds_home=row.get("odds_home", 0.0),
                    odds_draw=row.get("odds_draw", 0.0),
                    odds_away=row.get("odds_away", 0.0),
                    odds_over25=row.get("odds_over25", 0.0),
                )

                # Idempotencia: una cuota por partido + bookmaker
                with self._engine.connect() as conn:
                    existing = conn.execute(
                        text(
                            "SELECT [odds_id] "
                            "FROM [mundial].[FACT_ODDS] "
                            "WHERE [match_id] = :mid "
                            "AND [bookmaker] = :bk"
                        ),
                        {"mid": match_id, "bk": bookmaker},
                    ).scalar()

                if existing is not None:
                    skipped += 1
                    continue

                with get_session() as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO [mundial].[FACT_ODDS]
                            ([match_id], [bookmaker], [odds_home],
                             [odds_draw], [odds_away], [odds_over25])
                            VALUES (:match_id, :bookmaker, :odds_home,
                                    :odds_draw, :odds_away, :odds_over25)
                            """
                        ),
                        validated.to_db_dict(),
                    )
                inserted += 1

            except Exception as e:
                log.warning("Error cargando odds", fila=idx, error=str(e))
                errors += 1

        result = {"inserted": inserted, "skipped": skipped, "errors": errors}
        log.info("Carga de odds completada", **result)
        return result

    # ================================================================== #
    #  UTILIDADES                                                          #
    # ================================================================== #

    def get_table_counts(self) -> dict[str, int]:
        """Retorna el conteo de filas de todas las tablas del esquema.

        Returns
        -------
        dict[str, int]
            ``{tabla: conteo_filas}`` para verificación post-carga.
        """
        tables = [
            "DIM_TEAM",
            "DIM_COMPETITION",
            "FACT_MATCH",
            "FACT_ELO_HISTORY",
            "FACT_FIFA_RATING",
            "FACT_MATCH_XG",
            "FACT_ODDS",
            "FACT_PREDICTIONS",
            "ML_EXPERIMENT",
            "ML_MODEL_VERSION",
        ]
        counts: dict[str, int] = {}

        with self._engine.connect() as conn:
            for table in tables:
                try:
                    result = conn.execute(
                        text(
                            f"SELECT COUNT(*) FROM [mundial].[{table}]"
                        )
                    ).scalar()
                    counts[table] = result or 0
                except Exception:
                    counts[table] = -1  # Tabla no existe

        return counts

    def clear_caches(self) -> None:
        """Limpia los cachés de lookups para forzar re-lectura de BD."""
        self._team_cache.clear()
        self._competition_cache.clear()
        log.debug("Cachés de DataLoader limpiados")
