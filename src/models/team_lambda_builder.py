"""
TeamLambdaBuilder — El "eslabón perdido" entre el modelo y la simulación.

Convierte un modelo XGBoost entrenado en las **fuerzas base** (team_lambdas)
que consume el ``BracketEngine`` / ``MonteCarloSimulator``:

    {nombre_equipo: (lambda_ataque, lambda_defensa)}

Procedimiento
-------------
1. Ejecuta el pipeline de feature engineering reteniendo los identificadores
   de equipo (``home_team_id``/``away_team_id``) para poder mapear cada fila a
   un equipo concreto.
2. Construye un *snapshot* por equipo: el vector de features más reciente desde
   su propia perspectiva (rolling form, Elo, xG, ratings FIFA, descanso, etc.).
3. Para cada equipo enfrenta su snapshot contra un rival "promedio" del dataset
   (la media de todos los snapshots) en dos escenarios (como local y como
   visitante) y pide al modelo los goles esperados.
   - ``a_i`` (ataque)  = promedio de goles que el equipo *marca*.
   - ``d_i`` (defensa) = promedio de goles que el equipo *concede*.
4. Normaliza dividiendo por ``sqrt(media_global)`` para que el producto
   multiplicativo del bracket (``ataque_local × defensa_visitante``) reproduzca
   en promedio el ritmo goleador real de la liga internacional.

Las columnas del vector sintético se alinean **exactamente** con
``model_home.get_booster().feature_names`` para garantizar que no haya
discrepancias de features entre entrenamiento e inferencia.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_trainer import XGBoostTrainer
from src.utils.constants import K_FACTOR

logger = structlog.get_logger(__name__)


class TeamLambdaBuilder:
    """Genera fuerzas base por equipo a partir de un modelo entrenado.

    Parámetros
    ----------
    trainer : XGBoostTrainer
        Entrenador con ambos modelos (local/visitante) ya cargados.
    engine : Engine | None
        Motor SQLAlchemy para el ``FeatureEngineer``. Se ignora si se pasa
        ``feature_engineer`` directamente.
    feature_engineer : FeatureEngineer | None
        Instancia opcional del ingeniero de features (útil para tests).
    """

    def __init__(
        self,
        trainer: XGBoostTrainer,
        engine: Optional[Engine] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.trainer = trainer
        self.fe = feature_engineer or FeatureEngineer(engine=engine)

        # Estado cacheado (se calcula una sola vez en _prepare)
        self._prepared: bool = False
        self._feature_names: list[str] = []
        self._canon_keys: set[str] = set()
        self._snapshots: dict[str, dict[str, float]] = {}
        self._baseline: dict[str, float] = {}

        # Peso de competición para un partido de Mundial (K máximo normalizado)
        self._wc_weight: float = max(K_FACTOR.values()) / max(K_FACTOR.values())

    # ================================================================== #
    #  PREPARACIÓN (features + snapshots)                                  #
    # ================================================================== #

    def _engineer_with_ids(self) -> pd.DataFrame:
        """Ejecuta el pipeline de features reteniendo IDs y nombres de equipo.

        Replica la cadena de :meth:`FeatureEngineer.build_features` pero sin
        descartar las columnas de identificación, necesarias para asociar
        cada fila a un equipo.
        """
        fe = self.fe
        df = fe.load_feature_store()
        if df is None or df.empty:
            raise ValueError(
                "El feature store está vacío. Ejecute la ingesta "
                "(`mundial-cli ingest`) antes de construir las lambdas."
            )

        df = fe.build_target(df)
        df = fe.compute_rolling_features(df)
        df = fe.compute_differential_features(df)
        df = fe.encode_competition_importance(df)
        df = fe.encode_confederations(df)
        df = fe.compute_context_features(df)
        df = fe.handle_sentinels(df)
        return df

    def _model_feature_names(self) -> list[str]:
        """Obtiene la lista exacta de columnas con las que se entrenó el modelo."""
        if self.trainer.model_home is None or self.trainer.model_away is None:
            raise RuntimeError(
                "El modelo no está entrenado/cargado. No se pueden derivar "
                "las team_lambdas."
            )
        names = self.trainer.model_home.get_booster().feature_names
        if not names:
            raise RuntimeError(
                "El modelo entrenado no expone nombres de features; "
                "reentrene con un DataFrame de pandas."
            )
        return list(names)

    def _build_snapshots(
        self, df: pd.DataFrame, canon_keys: set[str]
    ) -> dict[str, dict[str, float]]:
        """Construye el snapshot de features más reciente por equipo.

        Funde las perspectivas local y visitante, ordena por fecha y conserva
        la observación más reciente de cada equipo.
        """
        parts: list[pd.DataFrame] = []
        for side in ("home", "away"):
            prefix = f"{side}_"
            id_col = f"{side}_team_id"
            name_col = f"{side}_team_name"
            if id_col not in df.columns or name_col not in df.columns:
                continue

            feat_cols = [
                c
                for c in df.columns
                if c.startswith(prefix) and c[len(prefix):] in canon_keys
            ]
            sub = df[[id_col, name_col, "match_date"] + feat_cols].copy()
            rename = {id_col: "team_id", name_col: "team_name"}
            rename.update({c: c[len(prefix):] for c in feat_cols})
            sub = sub.rename(columns=rename)
            parts.append(sub)

        if not parts:
            return {}

        combined = pd.concat(parts, ignore_index=True).sort_values("match_date")
        latest = combined.drop_duplicates(subset="team_name", keep="last")

        snapshots: dict[str, dict[str, float]] = {}
        for _, row in latest.iterrows():
            name = str(row["team_name"])
            snapshots[name] = {
                k: float(row[k])
                for k in canon_keys
                if k in row and pd.notna(row[k])
            }

        return snapshots

    def _prepare(self) -> None:
        """Calcula features, snapshots y baseline una sola vez (cacheado)."""
        if self._prepared:
            return

        self._feature_names = self._model_feature_names()
        self._canon_keys = {
            f[5:] for f in self._feature_names if f.startswith("home_")
        } | {
            f[5:] for f in self._feature_names if f.startswith("away_")
        }

        df = self._engineer_with_ids()
        self._snapshots = self._build_snapshots(df, self._canon_keys)
        if not self._snapshots:
            raise ValueError(
                "No se pudieron construir snapshots de equipos desde el "
                "feature store."
            )

        # Rival "promedio": media de cada feature canónica sobre todos los equipos
        self._baseline = {
            k: float(
                np.mean([s.get(k, 0.0) for s in self._snapshots.values()])
            )
            for k in self._canon_keys
        }

        self._prepared = True
        logger.info(
            "team_lambda_builder_preparado",
            equipos_con_snapshot=len(self._snapshots),
            n_features=len(self._feature_names),
        )

    # ================================================================== #
    #  ENSAMBLADO DE FILAS SINTÉTICAS                                      #
    # ================================================================== #

    def _assemble_row(
        self,
        home_snap: dict[str, float],
        away_snap: dict[str, float],
        is_neutral: bool,
        is_knockout: bool,
    ) -> dict[str, float]:
        """Ensambla una fila de features para un enfrentamiento sintético.

        Construye exactamente las columnas que el modelo espera, recomputando
        las features diferenciales a partir de los dos snapshots.
        """
        bl = self._baseline
        row: dict[str, float] = {}

        for feat in self._feature_names:
            if feat == "elo_diff":
                row[feat] = home_snap.get("elo", 0.0) - away_snap.get("elo", 0.0)
            elif feat == "xg_diff":
                row[feat] = home_snap.get("rolling_xg", 0.0) - away_snap.get(
                    "rolling_xg", 0.0
                )
            elif feat == "form_diff":
                row[feat] = home_snap.get("rolling_form", 0.0) - away_snap.get(
                    "rolling_form", 0.0
                )
            elif feat == "fifa_attack_diff":
                row[feat] = home_snap.get("fifa_attack", 0.0) - away_snap.get(
                    "fifa_attack", 0.0
                )
            elif feat == "goals_diff":
                row[feat] = home_snap.get(
                    "rolling_goals_scored", 0.0
                ) - away_snap.get("rolling_goals_scored", 0.0)
            elif feat == "is_neutral":
                row[feat] = float(int(is_neutral))
            elif feat == "is_knockout":
                row[feat] = float(int(is_knockout))
            elif feat == "competition_weight":
                row[feat] = self._wc_weight
            elif feat.startswith("home_"):
                key = feat[5:]
                row[feat] = home_snap.get(key, bl.get(key, 0.0))
            elif feat.startswith("away_"):
                key = feat[5:]
                row[feat] = away_snap.get(key, bl.get(key, 0.0))
            else:
                row[feat] = bl.get(feat, 0.0)

        return row

    # ================================================================== #
    #  API PÚBLICA                                                         #
    # ================================================================== #

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
        is_knockout: bool = False,
    ) -> pd.DataFrame:
        """Construye el vector de features (1 fila) para un partido concreto.

        Usa los snapshots reales de ambos equipos. Si un equipo no tiene
        historial en el feature store, se usa el rival promedio como respaldo.

        Retorna
        -------
        pd.DataFrame
            DataFrame de una fila con las columnas alineadas al modelo.
        """
        self._prepare()

        home_snap = self._snapshots.get(home_team)
        away_snap = self._snapshots.get(away_team)

        if home_snap is None:
            logger.warning(
                "snapshot_no_encontrado_usando_baseline", equipo=home_team
            )
            home_snap = self._baseline
        if away_snap is None:
            logger.warning(
                "snapshot_no_encontrado_usando_baseline", equipo=away_team
            )
            away_snap = self._baseline

        row = self._assemble_row(home_snap, away_snap, is_neutral, is_knockout)
        return pd.DataFrame([row])[self._feature_names]

    def build(
        self, teams: Optional[list[str]] = None
    ) -> dict[str, tuple[float, float]]:
        """Genera las fuerzas base ``(ataque, defensa)`` por equipo.

        Parámetros
        ----------
        teams : list[str] | None
            Lista de nombres canónicos de equipos. Si es ``None`` se usan
            todos los equipos con snapshot disponible.

        Retorna
        -------
        dict[str, tuple[float, float]]
            ``{equipo: (lambda_ataque, lambda_defensa)}`` normalizadas.
        """
        self._prepare()

        target_teams: list[str] = teams or list(self._snapshots.keys())
        bl = self._baseline

        # Ensamblar 2 filas por equipo (como local y como visitante vs baseline)
        rows: list[dict[str, float]] = []
        meta: list[tuple[str, str]] = []
        for team in target_teams:
            snap = self._snapshots.get(team)
            if snap is None:
                logger.warning(
                    "equipo_sin_historial_usando_baseline", equipo=team
                )
                snap = bl

            # Escenario neutral (Mundial), fase de grupos como referencia base
            rows.append(self._assemble_row(snap, bl, is_neutral=True, is_knockout=False))
            meta.append((team, "home"))
            rows.append(self._assemble_row(bl, snap, is_neutral=True, is_knockout=False))
            meta.append((team, "away"))

        X = pd.DataFrame(rows)[self._feature_names]
        lambda_home, lambda_away = self.trainer.predict(X)

        # Acumular goles marcados/concedidos por equipo
        scored: dict[str, list[float]] = {}
        conceded: dict[str, list[float]] = {}
        for i, (team, role) in enumerate(meta):
            if role == "home":
                scored.setdefault(team, []).append(float(lambda_home[i]))
                conceded.setdefault(team, []).append(float(lambda_away[i]))
            else:  # role == "away"
                scored.setdefault(team, []).append(float(lambda_away[i]))
                conceded.setdefault(team, []).append(float(lambda_home[i]))

        attack = {t: float(np.mean(v)) for t, v in scored.items()}
        defense = {t: float(np.mean(v)) for t, v in conceded.items()}

        # Normalizar para que el producto ataque×defensa reproduzca el ritmo medio
        mu = float(np.mean(list(attack.values()))) if attack else 1.0
        scale = math.sqrt(mu) if mu > 0 else 1.0

        team_lambdas: dict[str, tuple[float, float]] = {
            t: (attack[t] / scale, defense[t] / scale) for t in attack
        }

        logger.info(
            "team_lambdas_generadas",
            n_equipos=len(team_lambdas),
            ritmo_medio=round(mu, 3),
            ejemplo={
                t: (round(a, 3), round(d, 3))
                for t, (a, d) in list(team_lambdas.items())[:3]
            },
        )

        return team_lambdas
