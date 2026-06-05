"""
Módulo de Ingeniería de Features para el predictor del Mundial FIFA 2026.

Transforma los datos crudos de la vista ``mundial.vw_feature_store`` en un
DataFrame listo para entrenamiento, calculando ventanas rodantes por equipo,
diferencias home-away, codificación de competición y variables contextuales.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.constants import (
    CONFEDERATIONS,
    ELO_MOMENTUM_MONTHS,
    K_FACTOR,
    ROLLING_WINDOW_MATCHES,
    SENTINEL_FLOAT,
    SENTINEL_INT,
)
from src.utils.db import get_engine

logger = structlog.get_logger(__name__)

# ── Columnas que NO son features y se eliminan al final ──────────────────────
_NON_FEATURE_COLS: list[str] = [
    "match_id",
    "match_date",
    "competition_id",
    "competition_name",
    "stage",
    "home_team_id",
    "home_team_name",
    "home_fifa_code",
    "away_team_id",
    "away_team_name",
    "away_fifa_code",
    "venue",
    "home_goals",
    "away_goals",
    "ingested_at",
    "target_result",
]

# ── Columnas enteras donde -1 es centinela ───────────────────────────────────
_SENTINEL_INT_COLS: list[str] = [
    "attendance",
    "home_squad_size",
    "away_squad_size",
    "home_total_caps",
    "away_total_caps",
]


class FeatureEngineer:
    """
    Orquestador de ingeniería de features.

    Lee de la vista ``vw_feature_store``, calcula rolling windows por equipo,
    diferencias home-away, codifica competiciones y confederaciones, construye
    targets y devuelve un DataFrame listo para XGBoost.
    """

    def __init__(self, engine: Optional[Engine] = None) -> None:
        """
        Inicializa el ingeniero de features.

        Parámetros
        ----------
        engine : Engine, opcional
            Motor SQLAlchemy. Si es ``None`` se obtiene vía ``get_engine()``.
        """
        self._engine: Engine = engine or get_engine()
        self._cached_df: Optional[pd.DataFrame] = None
        logger.info("feature_engineer_inicializado", engine=str(self._engine.url))

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Carga de datos
    # ─────────────────────────────────────────────────────────────────────────
    def load_feature_store(self) -> pd.DataFrame:
        """
        Lee la vista ``mundial.vw_feature_store`` y la cachea en memoria.

        Retorna
        -------
        pd.DataFrame
            DataFrame completo de la vista con columnas originales.
        """
        if self._cached_df is not None:
            logger.debug("feature_store_cacheado", filas=len(self._cached_df))
            return self._cached_df.copy()

        query = text("SELECT * FROM mundial.vw_feature_store ORDER BY match_date")
        logger.info("cargando_feature_store")
        with self._engine.connect() as conn:
            self._cached_df = pd.read_sql(query, conn, parse_dates=["match_date"])

        logger.info(
            "feature_store_cargado",
            filas=len(self._cached_df),
            columnas=list(self._cached_df.columns),
        )
        return self._cached_df.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Rolling features (ventanas rodantes por equipo)
    # ─────────────────────────────────────────────────────────────────────────
    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features rodantes por equipo sobre los últimos
        ``ROLLING_WINDOW_MATCHES`` partidos.

        Procedimiento:
        1. *Melt* del DataFrame: cada partido genera 2 filas (perspectiva
           home y away) para rastrear el historial de cada equipo.
        2. Se calculan rolling means de goles, xG, xGA y porcentaje de
           victorias (forma).
        3. Se calcula el impulso Elo como la variación en los últimos
           ``ELO_MOMENTUM_MONTHS`` meses.
        4. Los stats se fusionan de vuelta al DataFrame a nivel de partido.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con columnas de la vista.

        Retorna
        -------
        pd.DataFrame
            DataFrame original enriquecido con columnas ``home_rolling_*``
            y ``away_rolling_*``.
        """
        df = df.copy()
        df = df.sort_values("match_date").reset_index(drop=True)

        # ── Perspectiva HOME ─────────────────────────────────────────────
        home_records = df[
            ["match_id", "match_date", "home_team_id", "home_goals", "away_goals",
             "home_xg", "home_xga", "home_elo", "home_elo_delta"]
        ].rename(columns={
            "home_team_id": "team_id",
            "home_goals": "goals_scored",
            "away_goals": "goals_conceded",
            "home_xg": "xg",
            "home_xga": "xga",
            "home_elo": "elo",
            "home_elo_delta": "elo_delta",
        })
        home_records["is_home"] = True

        # ── Perspectiva AWAY ─────────────────────────────────────────────
        away_records = df[
            ["match_id", "match_date", "away_team_id", "away_goals", "home_goals",
             "away_xg", "away_xga", "away_elo", "away_elo_delta"]
        ].rename(columns={
            "away_team_id": "team_id",
            "away_goals": "goals_scored",
            "home_goals": "goals_conceded",
            "away_xg": "xg",
            "away_xga": "xga",
            "away_elo": "elo",
            "away_elo_delta": "elo_delta",
        })
        away_records["is_home"] = False

        # ── Unión y ordenamiento cronológico por equipo ──────────────────
        team_history = (
            pd.concat([home_records, away_records], ignore_index=True)
            .sort_values(["team_id", "match_date"])
            .reset_index(drop=True)
        )

        # Victoria: 1 si ganó, 0 en otro caso
        team_history["win"] = (
            team_history["goals_scored"] > team_history["goals_conceded"]
        ).astype(float)

        # ── Rolling window (min_periods=1 para no perder filas iniciales) ─
        window = ROLLING_WINDOW_MATCHES
        grouped = team_history.groupby("team_id")

        team_history["rolling_goals_scored"] = grouped["goals_scored"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_history["rolling_goals_conceded"] = grouped["goals_conceded"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_history["rolling_xg"] = grouped["xg"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_history["rolling_xga"] = grouped["xga"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_history["rolling_form"] = grouped["win"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

        # ── Elo momentum (variación en últimos N meses) ──────────────────
        team_history["rolling_elo_momentum"] = grouped["elo_delta"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )

        rolling_cols = [
            "rolling_goals_scored",
            "rolling_goals_conceded",
            "rolling_xg",
            "rolling_xga",
            "rolling_form",
            "rolling_elo_momentum",
        ]

        # ── Separar home y away para merge ───────────────────────────────
        home_rolling = (
            team_history.loc[team_history["is_home"], ["match_id"] + rolling_cols]
            .rename(columns={c: f"home_{c}" for c in rolling_cols})
        )
        away_rolling = (
            team_history.loc[~team_history["is_home"], ["match_id"] + rolling_cols]
            .rename(columns={c: f"away_{c}" for c in rolling_cols})
        )

        df = df.merge(home_rolling, on="match_id", how="left")
        df = df.merge(away_rolling, on="match_id", how="left")

        logger.info(
            "rolling_features_calculadas",
            ventana=window,
            nuevas_columnas=[f"home_{c}" for c in rolling_cols]
            + [f"away_{c}" for c in rolling_cols],
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Features diferenciales (home − away)
    # ─────────────────────────────────────────────────────────────────────────
    def compute_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea diferencias home − away para las métricas clave.

        Columnas generadas:
        - ``xg_diff``: rolling xG home − rolling xG away
        - ``form_diff``: rolling form home − rolling form away
        - ``squad_value_diff``: valor de plantilla (log) home − away
        - ``goals_diff``: rolling goles anotados home − away
        - ``elo_diff`` ya existe en la vista y se preserva.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con columnas rolling ya calculadas.

        Retorna
        -------
        pd.DataFrame
            DataFrame con columnas diferenciales añadidas.
        """
        df = df.copy()

        df["xg_diff"] = df["home_rolling_xg"] - df["away_rolling_xg"]
        df["form_diff"] = df["home_rolling_form"] - df["away_rolling_form"]
        df["squad_value_diff"] = (
            df["home_squad_value_log"] - df["away_squad_value_log"]
        )
        df["goals_diff"] = (
            df["home_rolling_goals_scored"] - df["away_rolling_goals_scored"]
        )

        logger.info(
            "features_diferenciales_calculadas",
            columnas=["xg_diff", "form_diff", "squad_value_diff", "goals_diff"],
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Codificación de importancia de competición
    # ─────────────────────────────────────────────────────────────────────────
    def encode_competition_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mapea ``competition_name`` a un peso normalizado usando ``K_FACTOR``.

        El peso se normaliza dividiendo el valor K por el máximo (40),
        produciendo un rango [0.5, 1.0].  Si la competición no se encuentra
        en el diccionario se asigna el peso de ``friendly`` (0.5).

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con columna ``competition_name``.

        Retorna
        -------
        pd.DataFrame
            DataFrame con columna ``competition_weight`` añadida.
        """
        df = df.copy()
        max_k = max(K_FACTOR.values())

        # Normalizar el diccionario K_FACTOR
        k_normalized: dict[str, float] = {
            key: value / max_k for key, value in K_FACTOR.items()
        }
        default_weight = k_normalized.get("friendly", 0.5)

        def _map_competition(name: str) -> float:
            """Mapea nombre de competición a su peso normalizado."""
            if not isinstance(name, str):
                return default_weight
            name_lower = name.lower()
            for key, weight in k_normalized.items():
                if key.replace("_", " ") in name_lower or key in name_lower:
                    return weight
            return default_weight

        df["competition_weight"] = df["competition_name"].apply(_map_competition)

        logger.info(
            "competicion_codificada",
            pesos_unicos=df["competition_weight"].nunique(),
            distribucion=df["competition_weight"].value_counts().to_dict(),
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Codificación de confederaciones (placeholder)
    # ─────────────────────────────────────────────────────────────────────────
    def encode_confederations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea columnas one-hot placeholder para confederaciones home y away.

        La vista ``vw_feature_store`` no incluye confederación directamente,
        por lo que se crean columnas inicializadas en 0 que pueden poblarse
        posteriormente vía join con ``DIM_TEAM``.

        Columnas generadas:
        - ``home_conf_{confederación}`` para cada confederación
        - ``away_conf_{confederación}`` para cada confederación

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame de entrada.

        Retorna
        -------
        pd.DataFrame
            DataFrame con columnas de confederación placeholder.
        """
        df = df.copy()

        for conf in CONFEDERATIONS:
            df[f"home_conf_{conf}"] = 0
            df[f"away_conf_{conf}"] = 0

        logger.info(
            "confederaciones_codificadas",
            confederaciones=CONFEDERATIONS,
            columnas_creadas=len(CONFEDERATIONS) * 2,
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Features contextuales
    # ─────────────────────────────────────────────────────────────────────────
    def compute_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features de contexto del partido.

        - ``home_days_rest``: días desde el último partido del equipo local.
        - ``away_days_rest``: días desde el último partido del equipo visitante.
        - ``is_neutral`` e ``is_knockout`` ya están presentes.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con ``match_date``, ``home_team_id``, ``away_team_id``.

        Retorna
        -------
        pd.DataFrame
            DataFrame con columnas de descanso añadidas.
        """
        df = df.copy()
        df = df.sort_values("match_date").reset_index(drop=True)

        # ── Construir historial por equipo para calcular días de descanso ─
        home_dates = df[["match_id", "match_date", "home_team_id"]].rename(
            columns={"home_team_id": "team_id"}
        )
        away_dates = df[["match_id", "match_date", "away_team_id"]].rename(
            columns={"away_team_id": "team_id"}
        )
        all_dates = (
            pd.concat([home_dates, away_dates], ignore_index=True)
            .sort_values(["team_id", "match_date"])
            .reset_index(drop=True)
        )

        # Días desde el partido anterior de ese equipo
        all_dates["prev_date"] = all_dates.groupby("team_id")["match_date"].shift(1)
        all_dates["days_rest"] = (
            all_dates["match_date"] - all_dates["prev_date"]
        ).dt.days

        # Separar en home y away
        home_rest = (
            home_dates.merge(
                all_dates[["match_id", "team_id", "days_rest"]],
                on=["match_id", "team_id"],
                how="left",
            )
            .rename(columns={"days_rest": "home_days_rest"})
            [["match_id", "home_days_rest"]]
        )
        away_rest = (
            away_dates.merge(
                all_dates[["match_id", "team_id", "days_rest"]],
                on=["match_id", "team_id"],
                how="left",
            )
            .rename(columns={"days_rest": "away_days_rest"})
            [["match_id", "away_days_rest"]]
        )

        df = df.merge(home_rest, on="match_id", how="left")
        df = df.merge(away_rest, on="match_id", how="left")

        # Imputar primer partido de cada equipo con mediana global
        median_rest = df[["home_days_rest", "away_days_rest"]].median().mean()
        df["home_days_rest"] = df["home_days_rest"].fillna(median_rest)
        df["away_days_rest"] = df["away_days_rest"].fillna(median_rest)

        logger.info(
            "features_contextuales_calculadas",
            mediana_descanso=median_rest,
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Construcción de targets
    # ─────────────────────────────────────────────────────────────────────────
    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construye la variable objetivo para clasificación.

        - ``target_result``: 2 = victoria local, 1 = empate, 0 = victoria visitante.
        - ``home_goals`` y ``away_goals`` se mantienen como targets de regresión.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con ``home_goals`` y ``away_goals``.

        Retorna
        -------
        pd.DataFrame
            DataFrame con columna ``target_result`` añadida.
        """
        df = df.copy()

        conditions = [
            df["home_goals"] > df["away_goals"],   # Victoria local
            df["home_goals"] == df["away_goals"],   # Empate
            df["home_goals"] < df["away_goals"],    # Victoria visitante
        ]
        choices = [2, 1, 0]

        df["target_result"] = np.select(conditions, choices, default=1)

        logger.info(
            "targets_construidos",
            distribucion=df["target_result"].value_counts().to_dict(),
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Manejo de centinelas
    # ─────────────────────────────────────────────────────────────────────────
    def handle_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores centinela con ``np.nan`` e imputa con la mediana.

        Reglas:
        - Columnas enteras (attendance, squad_size, total_caps): ``-1`` → NaN.
        - Los valores ``0.0`` en columnas como xG o Elo **NO** se tocan,
          pues pueden ser datos reales (equipo sin datos de xG disponibles).
        - Después de reemplazar centinelas, se imputa con la mediana de la
          columna.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame potencialmente contaminado con centinelas.

        Retorna
        -------
        pd.DataFrame
            DataFrame limpio con centinelas imputados.
        """
        df = df.copy()
        sentinels_replaced = 0

        # ── Columnas enteras con centinela -1 ────────────────────────────
        for col in _SENTINEL_INT_COLS:
            if col in df.columns:
                mask = df[col] == SENTINEL_INT
                count = mask.sum()
                if count > 0:
                    df.loc[mask, col] = np.nan
                    sentinels_replaced += count
                    logger.debug(
                        "centinela_int_reemplazado",
                        columna=col,
                        cantidad=int(count),
                    )

        # ── Imputar con mediana ──────────────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_nans = df[col].isna().sum()
            if n_nans > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(
                    "columna_imputada",
                    columna=col,
                    nans=int(n_nans),
                    mediana=float(median_val),
                )

        logger.info(
            "centinelas_procesados",
            centinelas_reemplazados=int(sentinels_replaced),
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Orquestador principal
    # ─────────────────────────────────────────────────────────────────────────
    def build_features(
        self, df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Pipeline completo de ingeniería de features.

        Encadena todos los pasos de transformación y devuelve la matriz de
        features ``X``, más los targets de regresión ``y_home_goals`` y
        ``y_away_goals``.

        Parámetros
        ----------
        df : pd.DataFrame, opcional
            DataFrame de entrada. Si es ``None``, se carga de la BD.

        Retorna
        -------
        tuple[pd.DataFrame, pd.Series, pd.Series]
            - ``X``: DataFrame con features numéricas sin NaN.
            - ``y_home_goals``: Serie de goles del equipo local.
            - ``y_away_goals``: Serie de goles del equipo visitante.
        """
        if df is None:
            df = self.load_feature_store()

        logger.info("pipeline_features_iniciado", filas=len(df))

        # ── Paso 1: Targets ──────────────────────────────────────────────
        df = self.build_target(df)

        # ── Paso 2: Rolling features ─────────────────────────────────────
        df = self.compute_rolling_features(df)

        # ── Paso 3: Features diferenciales ───────────────────────────────
        df = self.compute_differential_features(df)

        # ── Paso 4: Codificación de competición ──────────────────────────
        df = self.encode_competition_importance(df)

        # ── Paso 5: Confederaciones (placeholder) ────────────────────────
        df = self.encode_confederations(df)

        # ── Paso 6: Features de contexto ─────────────────────────────────
        df = self.compute_context_features(df)

        # ── Paso 7: Limpieza de centinelas ───────────────────────────────
        df = self.handle_sentinels(df)

        # ── Extraer targets antes de eliminar columnas ───────────────────
        y_home_goals: pd.Series = df["home_goals"].copy()
        y_away_goals: pd.Series = df["away_goals"].copy()

        # ── Eliminar columnas no-feature ─────────────────────────────────
        cols_to_drop = [c for c in _NON_FEATURE_COLS if c in df.columns]
        X = df.drop(columns=cols_to_drop)

        # Eliminar columnas no numéricas restantes (strings, fechas)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.debug("eliminando_columnas_no_numericas", columnas=non_numeric)
            X = X.drop(columns=non_numeric)

        logger.info(
            "pipeline_features_completado",
            features=X.shape[1],
            filas=X.shape[0],
            nans_restantes=int(X.isna().sum().sum()),
        )
        return X, y_home_goals, y_away_goals
