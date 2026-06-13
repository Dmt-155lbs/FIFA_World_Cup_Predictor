"""
Capa de acceso a datos del Dashboard — la fuente única de verdad.

Centraliza toda la conectividad real del dashboard de Streamlit:
  - Conexión a SQL Server (vía ``src.utils.db``)
  - Carga del modelo entrenado + pipeline de inferencia
  - Generación de predicciones y simulaciones en vivo (modelo-driven)
  - Lectura de métricas / versiones desde ``ML_MODEL_VERSION`` y MLflow

Diseño defensivo: cada función está envuelta para degradar con elegancia.
Si la BD o el modelo no están disponibles, se devuelve un resultado vacío
(o ``None``) y la UI muestra un aviso claro en vez de datos inventados.

Todas las funciones costosas usan ``st.cache_*`` para evitar recomputar en
cada interacción del usuario.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Ruta del modelo serializado (compartida con el contenedor `app` vía volumen)
MODEL_PATH = os.getenv("MODEL_PATH", "./models/best_model")

# Calibrador de probabilidades 1X2, persistido por `run_evaluation` junto al
# modelo. El dashboard lo usa para calcular el EV de los Value Bets con
# probabilidades CALIBRADAS (las λ crudas no se tocan).
CALIBRATOR_PATH = os.path.join(os.path.dirname(MODEL_PATH) or ".", "calibrator.pkl")


# ============================================================================ #
#  RECURSOS CACHEADOS (engine, pipeline, builder)                              #
# ============================================================================ #


@st.cache_resource(show_spinner=False)
def get_engine():
    """Motor SQLAlchemy hacia SQL Server (singleton del proceso Streamlit)."""
    from src.utils.db import get_engine as _get_engine

    return _get_engine()


@st.cache_resource(show_spinner="Cargando modelo entrenado…")
def get_pipeline():
    """Carga el ``PredictionPipeline`` con el modelo entrenado desde disco.

    Devuelve ``None`` si el modelo no está disponible o si falta alguna
    dependencia pesada (se registra el error pero la UI sigue operativa con
    los datos de la BD).
    """
    try:
        from src.pipeline import PredictionPipeline

        pipeline = PredictionPipeline(model_path=MODEL_PATH)
        if pipeline.trainer.model_home is None:
            return None
        return pipeline
    except Exception as exc:  # pragma: no cover - depende del entorno
        logger.warning("No se pudo cargar el pipeline: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_lambda_builder():
    """Crea el ``TeamLambdaBuilder`` reutilizando el pipeline cargado."""
    pipeline = get_pipeline()
    if pipeline is None:
        return None
    try:
        from src.models.team_lambda_builder import TeamLambdaBuilder

        return TeamLambdaBuilder(
            trainer=pipeline.trainer,
            feature_engineer=pipeline.feature_engineer,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("No se pudo crear el TeamLambdaBuilder: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_calibrator():
    """Carga el calibrador de probabilidades 1X2 persistido por la evaluación.

    Devuelve ``None`` si aún no se ha generado (ejecutar ``mundial-cli
    evaluate``) o si no se puede cargar, en cuyo caso el dashboard cae con
    elegancia a las probabilidades crudas.
    """
    if not os.path.exists(CALIBRATOR_PATH):
        return None
    try:
        from src.models.probability_calibrator import ProbabilityCalibrator

        calib = ProbabilityCalibrator.load(CALIBRATOR_PATH)
        return calib if calib.fitted else None
    except Exception as exc:  # pragma: no cover
        logger.warning("No se pudo cargar el calibrador: %s", exc)
        return None


# ============================================================================ #
#  ESTADO DE CONEXIÓN                                                           #
# ============================================================================ #


@st.cache_data(ttl=60, show_spinner=False)
def db_available() -> bool:
    """Comprueba si la conexión a la base de datos responde."""
    try:
        from src.utils.db import test_connection

        return test_connection()
    except Exception:
        return False


def model_available() -> bool:
    """Indica si el modelo entrenado está cargado y listo para inferencia."""
    return get_pipeline() is not None


# ============================================================================ #
#  CATÁLOGOS Y MÉTRICAS                                                          #
# ============================================================================ #


@st.cache_data(ttl=300, show_spinner=False)
def list_teams() -> list[str]:
    """Los 48 equipos del Mundial 2026 (nombre canónico) ordenados por ranking FIFA.

    Lee de ``DIM_TEAM`` pero filtra a los 48 clasificados: la tabla también
    contiene selecciones históricas (Italia, Ucrania, etc.) que existen sólo
    como datos de entrenamiento de ``FACT_MATCH`` y NO deben aparecer en los
    selectores del torneo. Si la BD no responde, cae en el seed estático.
    """
    from src.ingestion.seed_data import WORLD_CUP_2026_TEAMS

    wc_teams = {t[0] for t in WORLD_CUP_2026_TEAMS}
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT [team_name] FROM [mundial].[DIM_TEAM] "
                    "ORDER BY [fifa_ranking]"
                )
            ).fetchall()
        names = [r[0] for r in rows if r[0] in wc_teams]
        if names:
            return names
    except Exception:
        pass

    # Orden por ranking del seed
    return [t[0] for t in sorted(WORLD_CUP_2026_TEAMS, key=lambda x: x[3])]


@st.cache_data(ttl=300, show_spinner=False)
def get_model_versions() -> pd.DataFrame:
    """Versiones de modelo registradas en ``ML_MODEL_VERSION`` (+ experimento).

    Retorna un DataFrame (posiblemente vacío) ordenado de más reciente a más
    antiguo.
    """
    query = text(
        """
        SELECT mv.[version_tag],
               mv.[brier_score],
               mv.[roi_backtest],
               mv.[log_loss],
               mv.[artifact_path],
               mv.[trained_at],
               e.[experiment_name],
               e.[mlflow_run_id]
        FROM [mundial].[ML_MODEL_VERSION] AS mv
        INNER JOIN [mundial].[ML_EXPERIMENT] AS e
            ON mv.[experiment_id] = e.[experiment_id]
        ORDER BY mv.[trained_at] DESC
        """
    )
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame(
            columns=[
                "version_tag", "brier_score", "roi_backtest", "log_loss",
                "artifact_path", "trained_at", "experiment_name",
                "mlflow_run_id",
            ]
        )


def get_latest_metrics() -> dict[str, Any]:
    """Métricas de la versión de modelo más reciente (o ``{}`` si no hay)."""
    df = get_model_versions()
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


# ============================================================================ #
#  PREDICCIÓN DE PARTIDO (en vivo, modelo-driven)                              #
# ============================================================================ #


@st.cache_data(ttl=600, show_spinner="Calculando predicción…")
def predict_match(home_team: str, away_team: str) -> Optional[dict[str, Any]]:
    """Genera la predicción completa de un cruce usando el modelo real.

    Retorna un diccionario con probabilidades 1X2, lambdas, goles esperados
    y la matriz de marcadores de Poisson (para el heatmap), o ``None`` si el
    modelo no está disponible.
    """
    pipeline = get_pipeline()
    builder = get_lambda_builder()
    if pipeline is None or builder is None:
        return None

    try:
        features = builder.build_match_features(home_team, away_team)
        lambda_home_arr, lambda_away_arr = pipeline.trainer.predict(features)
        lambda_home = float(lambda_home_arr[0])
        lambda_away = float(lambda_away_arr[0])

        matrix = pipeline.poisson.score_matrix(lambda_home, lambda_away)
        probs = pipeline.poisson.match_probabilities(matrix)
        ou = pipeline.poisson.over_under_probs(matrix)
        exp_goals = pipeline.poisson.expected_goals(matrix)
        top_scores = pipeline.poisson.most_likely_scores(matrix)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "prob_home": probs["prob_home"],
            "prob_draw": probs["prob_draw"],
            "prob_away": probs["prob_away"],
            "prob_over_25": ou["prob_over"],
            "prob_under_25": ou["prob_under"],
            "expected_home_goals": float(exp_goals[0]),
            "expected_away_goals": float(exp_goals[1]),
            "most_likely_scores": top_scores,
            "score_matrix": np.asarray(matrix),
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("Fallo al predecir el partido: %s", exc)
        return None


# ============================================================================ #
#  SIMULACIÓN MONTE CARLO (en vivo)                                            #
# ============================================================================ #


@st.cache_data(ttl=1800, show_spinner="Ejecutando simulación Monte Carlo…")
def run_simulation(n_simulations: int = 10000) -> Optional[dict[str, Any]]:
    """Ejecuta la simulación completa del torneo con las team_lambdas reales.

    Retorna un dict serializable con las probabilidades agregadas, o ``None``
    si el modelo no está disponible.
    """
    pipeline = get_pipeline()
    builder = get_lambda_builder()
    if pipeline is None or builder is None:
        return None

    try:
        from src.ingestion.seed_data import WORLD_CUP_2026_TEAMS

        team_names = [t[0] for t in WORLD_CUP_2026_TEAMS]
        team_lambdas = builder.build(teams=team_names)
        results = pipeline.simulate_tournament(
            team_lambdas, n_simulations=n_simulations
        )

        return {
            "n_simulations": results.n_simulations,
            "champion_probs": dict(results.champion_probs),
            "finalist_probs": dict(results.finalist_probs),
            "semifinalist_probs": dict(results.semifinalist_probs),
            "round_advance_probs": {
                k: dict(v) for k, v in results.round_advance_probs.items()
            },
            "team_lambdas": {t: list(v) for t, v in team_lambdas.items()},
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("Fallo en la simulación: %s", exc)
        return None


# ============================================================================ #
#  BACKTESTING / VALUE BETS                                                     #
# ============================================================================ #


@st.cache_data(ttl=600, show_spinner="Cruzando el modelo con las casas de apuestas…")
def get_value_bets(ev_threshold: float = 0.05) -> pd.DataFrame:
    """Detecta value bets en partidos próximos cruzando cuotas y modelo.

    Lee las cuotas de ``FACT_ODDS`` de partidos FUTUROS (fixtures con
    ``is_played = 0`` cargados desde el feed en vivo de The Odds API), calcula
    la probabilidad del modelo para cada cruce y devuelve las selecciones cuyo
    Expected Value (EV = p_modelo · cuota − 1) supera el umbral. ``Prob. Modelo``
    es la probabilidad **CALIBRADA** (si existe el calibrador persistido por la
    evaluación; corrige el optimismo del Poisson con los underdogs) y el EV se
    calcula con ella. El EV usa la cuota decimal real de las casas; ``Prob. Casa``
    muestra la probabilidad implícita del mercado sin el margen (de-vig) para
    comparar de forma justa contra la del modelo.

    Cada fila incluye una columna ``Fiabilidad`` (``Alta``/``Baja``) con dos
    guardias: (1) alguno de los equipos no tiene historial real (tras cargar los
    12 antes ciegos esto ya no ocurre en el Mundial); (2) el EV es implausible
    (> 35%, techo de un mercado líquido) o la prob del modelo dispara respecto a
    la del mercado (≥2× y ≥10 pp), señal de que el modelo Poisson sobre-estima al
    underdog en vez de una ventaja real. Las apuestas de alta fiabilidad se
    muestran primero.

    Retorna un DataFrame (posiblemente vacío) con columnas:
    ``Fecha, Partido, Pick, Cuota, Prob. Modelo, Prob. Casa, EV, Fiabilidad``,
    ordenado por fiabilidad y luego EV descendente.
    """
    cols = [
        "Fecha", "Partido", "Pick", "Cuota",
        "Prob. Modelo", "Prob. Casa", "EV", "Fiabilidad",
    ]

    # Equipos con snapshot REAL (historial de partidos). Si un equipo no está
    # aquí, build_match_features usa el baseline promedio → predicción poco
    # fiable para esa selección. Se usa para marcar la fiabilidad de cada bet.
    known_teams: set[str] = set()
    builder = get_lambda_builder()
    if builder is not None:
        try:
            builder._prepare()
            known_teams = set(builder._snapshots.keys())
        except Exception:
            known_teams = set()

    # Calibrador de probabilidades: si está disponible, el EV se calcula con las
    # probabilidades CALIBRADAS (suavizan el optimismo del modelo con underdogs).
    calibrator = get_calibrator()

    query = text(
        """
        SELECT ht.[team_name]  AS home_team,
               at.[team_name]  AS away_team,
               MIN(m.[match_date]) AS match_date,
               AVG(o.[odds_home]) AS odds_home,
               AVG(o.[odds_draw]) AS odds_draw,
               AVG(o.[odds_away]) AS odds_away
        FROM [mundial].[FACT_ODDS] AS o
        INNER JOIN [mundial].[FACT_MATCH] AS m ON o.[match_id] = m.[match_id]
        INNER JOIN [mundial].[DIM_TEAM] AS ht ON m.[home_team_id] = ht.[team_id]
        INNER JOIN [mundial].[DIM_TEAM] AS at ON m.[away_team_id] = at.[team_id]
        WHERE m.[match_date] >= :today
          AND o.[odds_home] > 0 AND o.[odds_draw] > 0 AND o.[odds_away] > 0
        GROUP BY ht.[team_name], at.[team_name]
        """
    )
    try:
        with get_engine().connect() as conn:
            fixtures = pd.read_sql(
                query, conn, params={"today": date.today()},
                parse_dates=["match_date"],
            )
    except Exception:
        return pd.DataFrame(columns=cols)

    if fixtures.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    label = {"home": "Victoria Local", "draw": "Empate", "away": "Victoria Visitante"}
    for _, fx in fixtures.iterrows():
        pred = predict_match(fx["home_team"], fx["away_team"])
        if pred is None:
            continue
        # Probabilidades del modelo: CALIBRADAS si hay calibrador, crudas si no.
        # (Las λ crudas de pred siguen disponibles para otras páginas; aquí solo
        # transformamos las probabilidades 1X2 usadas para el EV financiero.)
        if calibrator is not None:
            ch, cd, ca = calibrator.predict_one(
                pred["prob_home"], pred["prob_draw"], pred["prob_away"]
            )
            model_probs = {"home": ch, "draw": cd, "away": ca}
        else:
            model_probs = {
                "home": pred["prob_home"],
                "draw": pred["prob_draw"],
                "away": pred["prob_away"],
            }
        odds = {
            "home": float(fx["odds_home"]),
            "draw": float(fx["odds_draw"]),
            "away": float(fx["odds_away"]),
        }
        # Probabilidad implícita del mercado sin margen (de-vig): normaliza
        # 1/cuota por la suma de las tres para descontar el overround.
        inv = {k: 1.0 / v for k, v in odds.items() if v > 0}
        overround = sum(inv.values()) or 1.0
        market_probs = {k: inv.get(k, 0.0) / overround for k in odds}

        fecha = (
            fx["match_date"].strftime("%d %b")
            if pd.notna(fx["match_date"]) else "—"
        )
        # Guardia 1: ambos equipos deben tener historial real (tras cargar los
        # 12 antes ciegos, esto ya se cumple para todo el Mundial; se mantiene
        # como red de seguridad).
        has_history = (
            (not known_teams)
            or (fx["home_team"] in known_teams and fx["away_team"] in known_teams)
        )

        for outcome in ("home", "draw", "away"):
            ev = model_probs[outcome] * odds[outcome] - 1.0
            if ev > ev_threshold:
                # Guardia 2: optimismo del modelo. El modelo Poisson sub-dispersa
                # los duelos desparejos (sub-rate al favorito, sobre-rate empate y
                # underdog); fix de datos (Elo + 12 equipos) NO lo corrige porque
                # es estructural. Un mercado líquido del Mundial casi nunca deja
                # ventajas > ~35%, así que un EV altísimo es casi siempre error de
                # calibración, no valor real. Marcamos baja confianza si el EV
                # supera ese techo O si la prob del modelo dispara respecto a la
                # del mercado (≥2× y ≥10 pp). Así el dashboard sólo destaca como
                # `Alta` las ventajas pequeñas y creíbles (Brazil/Belgium ~+8-10%).
                mkt = market_probs[outcome]
                blowout = (
                    ev > 0.35
                    or (
                        mkt > 0
                        and model_probs[outcome] >= 2.0 * mkt
                        and (model_probs[outcome] - mkt) >= 0.10
                    )
                )
                reliable = has_history and not blowout
                pick = fx["home_team"] if outcome == "home" else (
                    fx["away_team"] if outcome == "away" else "Empate"
                )
                rows.append({
                    "Fecha": fecha,
                    "Partido": f"{fx['home_team']} vs {fx['away_team']}",
                    "Pick": label[outcome] if outcome == "draw" else pick,
                    "Cuota": round(odds[outcome], 2),
                    "Prob. Modelo": f"{model_probs[outcome]:.1%}",
                    "Prob. Casa": f"{market_probs[outcome]:.1%}",
                    "EV": f"+{ev:.1%}",
                    "Fiabilidad": "Alta" if reliable else "Baja ⚠️",
                    "_ev": ev,
                    "_reliable": reliable,
                })

    if not rows:
        return pd.DataFrame(columns=cols)

    # Las apuestas de alta fiabilidad primero; dentro de cada grupo, mayor EV.
    out = pd.DataFrame(rows).sort_values(
        ["_reliable", "_ev"], ascending=[False, False]
    )
    return out[cols].reset_index(drop=True)


# ============================================================================ #
#  SHAP (artefactos de MLflow)                                                  #
# ============================================================================ #


@st.cache_data(ttl=600, show_spinner=False)
def get_shap_artifact(artifact_subpath: str = "shap_summary") -> Optional[str]:
    """Descarga un artefacto SHAP del run de MLflow más reciente.

    Retorna la ruta local del directorio/imagen descargada, o ``None`` si no
    hay run o MLflow no está accesible.
    """
    meta = get_latest_metrics()
    run_id = meta.get("mlflow_run_id")
    if not run_id or run_id == "UNKNOWN":
        return None

    try:
        import mlflow

        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_subpath
        )
        return local_path
    except Exception:
        return None
