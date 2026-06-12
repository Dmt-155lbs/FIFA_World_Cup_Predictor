"""
CLI Centralizado — El "Chef" del Sistema.

Punto de entrada unificado para interactuar con todas las capas del pipeline
(Ingesta, Entrenamiento, Simulación, Evaluación) mediante comandos de terminal.
Utiliza Typer para validación estricta y CLI documentation.

Uso:
    python -m src.cli --help
    python -m src.cli init-db
    python -m src.cli ingest --source fbref
    python -m src.cli train --optimize
"""

import sys
from datetime import date
from typing import Annotated, Optional

import structlog
import typer

from src.config import get_settings

logger = structlog.get_logger(__name__)

app = typer.Typer(
    name="mundial-cli",
    help="CLI Centralizado para el Predictor del Mundial FIFA 2026",
    add_completion=False,
)


@app.command("init-db")
def init_db() -> None:
    """Inicializa la base de datos (Ejecuta schemas y puebla dimensiones)."""
    from src.ingestion.db_init import init_database
    
    try:
        init_database()
        typer.secho("✅ Base de datos inicializada correctamente.", fg=typer.colors.GREEN)
    except Exception as e:
        logger.error("Error inicializando BD", error=str(e))
        typer.secho(f"❌ Error inicializando BD: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("seed")
def seed() -> None:
    """Puebla las dimensiones DIM_TEAM y DIM_COMPETITION con datos semilla."""
    from src.ingestion.loader import DataLoader
    from src.ingestion.seed_data import COMPETITIONS, WORLD_CUP_2026_TEAMS

    typer.secho("🌱 Poblando dimensiones (teams + competitions)...", fg=typer.colors.BLUE)
    try:
        loader = DataLoader()
        teams_inserted = loader.seed_teams(WORLD_CUP_2026_TEAMS)
        comps_inserted = loader.seed_competitions(COMPETITIONS)
        typer.secho(
            f"✅ Seed completado: {teams_inserted} equipos, "
            f"{comps_inserted} competiciones insertadas.",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        logger.error("Error durante seed", error=str(e))
        typer.secho(f"❌ Error durante seed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("ingest")
def ingest(
    source: Annotated[
        Optional[str],
        typer.Option(
            "--source", "-s",
            help="Fuente a ingerir: 'fbref', 'elo', 'fifa', 'odds', 'all'.",
        )
    ] = "all",
    start_year: Annotated[
        Optional[int],
        typer.Option(help="Año de inicio para datos históricos.")
    ] = None,
) -> None:
    """Ejecuta los scrapers y carga los datos en la base de datos (FACT tables)."""
    from src.ingestion.loader import DataLoader
    from src.ingestion.fbref_scraper import FBrefScraper
    from src.ingestion.elo_scraper import EloScraper
    from src.ingestion.sofifa_scraper import SoFIFAScraper
    from src.ingestion.odds_scraper import OddsScraperLive

    settings = get_settings()
    year = start_year or settings.data_start_year
    
    loader = DataLoader()
    typer.secho(f"🚀 Iniciando ingesta desde {year} para source={source}", fg=typer.colors.BLUE)

    # Cada fuente se ejecuta de forma AISLADA: si una falla (rate-limit, cambio
    # de HTML, falta de credenciales, etc.) se registra y se continúa con las
    # demás, en lugar de abortar toda la ingesta. Así `--source all` siempre
    # carga lo que esté disponible.
    def _run_source(name: str, scraper_factory, load_fn) -> bool:
        if source not in [name, "all"]:
            return True
        typer.secho(f"➡️  Fuente '{name}'...", fg=typer.colors.CYAN)
        try:
            df = scraper_factory().fetch_with_retry()
            res = load_fn(df)
            typer.secho(
                f"{name}: Insertados={res['inserted']}, "
                f"Skipped={res['skipped']}, Errores={res['errors']}",
                fg=typer.colors.GREEN,
            )
            return True
        except Exception as e:
            logger.warning("Fuente de ingesta fallida", fuente=name, error=str(e))
            typer.secho(
                f"⚠️  Fuente '{name}' omitida ({type(e).__name__}: {e}). Continuando.",
                fg=typer.colors.YELLOW,
            )
            return False

    results_ok = [
        _run_source("elo", EloScraper, loader.load_elo_history),
        _run_source("fbref", FBrefScraper, loader.load_matches),
        _run_source("fifa", SoFIFAScraper, loader.load_fifa_ratings),
        _run_source("odds", OddsScraperLive, loader.load_odds),
    ]

    if all(results_ok):
        typer.secho("✅ Ingesta completada (todas las fuentes).", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "✅ Ingesta finalizada con fuentes parciales "
            "(ver advertencias arriba).",
            fg=typer.colors.GREEN,
        )


@app.command("train")
def train(
    optimize: Annotated[
        bool,
        typer.Option("--optimize/--no-optimize", help="Ejecutar Optuna HPO.")
    ] = True,
) -> None:
    """Entrena el modelo XGBoost y calibra la distribución de Poisson."""
    from src.pipeline import PredictionPipeline
    
    typer.secho("🧠 Iniciando entrenamiento del modelo...", fg=typer.colors.BLUE)
    try:
        pipeline = PredictionPipeline()
        metrics = pipeline.train(optimize_hyperparams=optimize)
        typer.secho("✅ Entrenamiento completado.", fg=typer.colors.GREEN)
        
        for k, v in metrics.items():
            typer.echo(f"  - {k}: {v:.4f}")
            
    except Exception as e:
        logger.error("Error durante entrenamiento", error=str(e))
        typer.secho(f"❌ Error durante entrenamiento: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("predict")
def predict_match(
    home: Annotated[str, typer.Argument(help="Nombre del equipo local")],
    away: Annotated[str, typer.Argument(help="Nombre del equipo visitante")],
) -> None:
    """Genera una predicción para un partido específico."""
    from src.pipeline import PredictionPipeline

    typer.secho(f"⚽ Prediciendo: {home} vs {away}...", fg=typer.colors.BLUE)

    try:
        from src.models.team_lambda_builder import TeamLambdaBuilder

        pipeline = PredictionPipeline(model_path='./models/best_model')

        # Construir el vector de features real para este enfrentamiento
        # a partir de los snapshots más recientes de cada equipo en la BD.
        builder = TeamLambdaBuilder(
            trainer=pipeline.trainer,
            feature_engineer=pipeline.feature_engineer,
        )
        df_features = builder.build_match_features(home, away)

        pred = pipeline.predict_match(df_features, home_team=home, away_team=away)
        
        typer.secho("📊 Resultados:", fg=typer.colors.GREEN)
        typer.echo(f"  Prob {home}: {pred.prob_home:.1%}")
        typer.echo(f"  Prob Empate: {pred.prob_draw:.1%}")
        typer.echo(f"  Prob {away}: {pred.prob_away:.1%}")
        typer.echo(f"  Goles Esperados: {home} {pred.expected_home_goals:.2f} - {pred.expected_away_goals:.2f} {away}")
        
    except Exception as e:
        logger.error("Error durante predicción", error=str(e))
        typer.secho(f"❌ Error durante predicción: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("simulate")
def simulate(
    n_sims: Annotated[int, typer.Option("--sims", help="Número de simulaciones")] = 10000,
) -> None:
    """Ejecuta una simulación Monte Carlo del torneo completo."""
    from src.pipeline import PredictionPipeline
    
    typer.secho(f"🎲 Iniciando {n_sims} simulaciones Monte Carlo...", fg=typer.colors.BLUE)
    try:
        from src.models.team_lambda_builder import TeamLambdaBuilder
        from src.ingestion.seed_data import WORLD_CUP_2026_TEAMS

        pipeline = PredictionPipeline(model_path='./models/best_model')

        # Derivar las fuerzas base reales de cada equipo desde el modelo entrenado.
        typer.secho("Derivando team_lambdas desde el modelo entrenado...", fg=typer.colors.CYAN)
        builder = TeamLambdaBuilder(
            trainer=pipeline.trainer,
            feature_engineer=pipeline.feature_engineer,
        )
        team_names = [team[0] for team in WORLD_CUP_2026_TEAMS]
        team_lambdas = builder.build(teams=team_names)

        results = pipeline.simulate_tournament(team_lambdas, n_simulations=n_sims)
        
        typer.secho("🏆 Top 5 Favoritos:", fg=typer.colors.GREEN)
        top5 = sorted(results.champion_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for idx, (team, prob) in enumerate(top5, 1):
            typer.echo(f"  {idx}. {team}: {prob:.1%}")
            
    except Exception as e:
        logger.error("Error durante simulación", error=str(e))
        typer.secho(f"❌ Error durante simulación: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate() -> None:
    """Ejecuta la evaluación offline (Walk-Forward)."""
    import subprocess
    typer.secho("📈 Iniciando evaluación offline...", fg=typer.colors.BLUE)
    try:
        # Re-use the existing run_evaluation script
        result = subprocess.run([sys.executable, "-m", "src.run_evaluation"], check=True)
        typer.secho("✅ Evaluación completada.", fg=typer.colors.GREEN)
    except subprocess.CalledProcessError as e:
        typer.secho(f"❌ Falló la evaluación con código {e.returncode}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("serve")
def serve() -> None:
    """Inicia el demonio de scheduling (ingesta + reentrenamiento + simulación)."""
    from src.scheduler import start_scheduler

    typer.secho("🛰️ Iniciando scheduler (Ctrl+C para detener)...", fg=typer.colors.BLUE)
    try:
        start_scheduler()
    except Exception as e:
        logger.error("Error en el scheduler", error=str(e))
        typer.secho(f"❌ Error en el scheduler: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("run-all")
def run_all() -> None:
    """Orquesta todo: init-db -> seed -> ingest -> train -> evaluate."""
    typer.secho("🚀 INICIANDO EJECUCIÓN END-TO-END", fg=typer.colors.MAGENTA, bold=True)

    typer.echo("\n--- 1. INICIALIZAR BD ---")
    init_db()

    typer.echo("\n--- 2. POBLAR DIMENSIONES (SEED) ---")
    seed()

    typer.echo("\n--- 3. INGESTA DE DATOS ---")
    ingest(source="all")

    typer.echo("\n--- 4. ENTRENAMIENTO ---")
    train(optimize=True)

    typer.echo("\n--- 5. EVALUACIÓN ---")
    evaluate()

    typer.secho("\n🎉 EJECUCIÓN END-TO-END COMPLETADA CON ÉXITO", fg=typer.colors.MAGENTA, bold=True)


if __name__ == "__main__":
    app()
