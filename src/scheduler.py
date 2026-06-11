import time
import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.pipeline import PredictionPipeline
from src.models.team_lambda_builder import TeamLambdaBuilder
from src.ingestion.seed_data import WORLD_CUP_2026_TEAMS
from src.utils.alerts import AlertManager
from src.ingestion.fbref_scraper import FBrefScraper
from src.ingestion.elo_scraper import EloScraper
from src.ingestion.odds_scraper import OddsScraperLive

logger = structlog.get_logger(__name__)


def run_ingestion_job():
    """Ejecuta la ingesta periódica de datos desde todas las fuentes."""
    logger.info("Ejecutando job de ingesta de datos")
    try:
        # FBRef — resultados y xG
        fbref = FBrefScraper()
        df_fbref = fbref.fetch_with_retry()
        logger.info("FBRef scraping completado", filas=len(df_fbref))

        # Elo ratings
        elo = EloScraper()
        df_elo = elo.fetch_with_retry()
        logger.info("Elo scraping completado", filas=len(df_elo))

        # Odds en vivo
        odds = OddsScraperLive()
        df_odds = odds.fetch_with_retry()
        logger.info("Odds scraping completado", filas=len(df_odds))

        logger.info("Ingesta completada exitosamente")
    except Exception as e:
        logger.error("Error en job de ingesta", error=str(e))
        AlertManager().send_alert(
            title="Fallo en Ingesta",
            message=f"Error durante la ingesta de datos: {str(e)}",
            level="critical"
        )


def run_retrain_job():
    """Ejecuta el reentrenamiento periódico del modelo."""
    logger.info("Ejecutando job de reentrenamiento")
    try:
        pipeline = PredictionPipeline()
        metrics = pipeline.train(optimize_hyperparams=False)

        # Usar poisson_nloglik como proxy de degradación
        # (XGBoostTrainer.train() retorna eval_poisson_nloglik_home/away)
        nloglik = metrics.get('eval_poisson_nloglik_home', 1.0)
        if nloglik > 1.0:
            AlertManager().send_alert(
                title="Degradación de Modelo",
                message=(
                    f"Poisson NLogLik crítico tras reentrenamiento: "
                    f"{nloglik:.4f}"
                ),
                level="critical"
            )
        else:
            AlertManager().send_alert(
                title="Reentrenamiento Exitoso",
                message=(
                    f"Eval NLogLik Home: "
                    f"{metrics.get('eval_poisson_nloglik_home', 'N/A')}, "
                    f"Away: "
                    f"{metrics.get('eval_poisson_nloglik_away', 'N/A')}"
                ),
                level="info"
            )
    except Exception as e:
        logger.error("Error en job de reentrenamiento", error=str(e))
        AlertManager().send_alert(
            title="Fallo en Reentrenamiento",
            message=f"Error: {str(e)}",
            level="critical"
        )


def run_simulation_job():
    """Ejecuta la re-simulación Monte Carlo del torneo."""
    logger.info("Ejecutando job de simulación Monte Carlo")
    try:
        pipeline = PredictionPipeline(model_path='./models/best_model')

        if pipeline.trainer.model_home is None:
            logger.warning(
                "No hay modelo entrenado disponible. "
                "Ejecute el reentrenamiento primero."
            )
            return

        # Derivar las team_lambdas desde el modelo entrenado vía
        # TeamLambdaBuilder y simular el torneo completo.
        builder = TeamLambdaBuilder(
            trainer=pipeline.trainer,
            feature_engineer=pipeline.feature_engineer,
        )
        team_names = [team[0] for team in WORLD_CUP_2026_TEAMS]
        team_lambdas = builder.build(teams=team_names)

        results = pipeline.simulate_tournament(team_lambdas)

        # Reportar los favoritos al título
        top5 = sorted(
            results.champion_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        logger.info(
            "Simulación Monte Carlo completada",
            favoritos={team: round(prob, 4) for team, prob in top5},
        )

    except Exception as e:
        logger.error("Error en job de simulación", error=str(e))
        AlertManager().send_alert(
            title="Fallo en Simulación",
            message=f"Error: {str(e)}",
            level="critical"
        )


def start_scheduler():
    """Inicia el scheduler con los 3 jobs programados."""
    logger.info("Iniciando Scheduler de Predicción Mundial")
    scheduler = BackgroundScheduler()

    # Ingesta post-partido: cada 4 horas
    scheduler.add_job(
        func=run_ingestion_job,
        trigger=CronTrigger(hour='*/4'),
        id='ingestion_job',
        name='Ingesta de resultados',
        replace_existing=True
    )

    # Reentrenamiento diario a las 02:00 UTC
    scheduler.add_job(
        func=run_retrain_job,
        trigger=CronTrigger(hour=2, minute=0),
        id='retrain_job',
        name='Reentrenamiento del ensemble',
        replace_existing=True
    )

    # Re-simulación Monte Carlo pre-ronda a las 06:00 UTC
    scheduler.add_job(
        func=run_simulation_job,
        trigger=CronTrigger(hour=6, minute=0),
        id='simulation_job',
        name='Re-simulación Monte Carlo',
        replace_existing=True
    )

    scheduler.start()

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Apagando scheduler...")
        scheduler.shutdown()


if __name__ == "__main__":
    start_scheduler()
