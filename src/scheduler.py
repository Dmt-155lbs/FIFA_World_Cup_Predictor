import time
import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.pipeline import PredictionPipeline
from src.utils.alerts import AlertManager
from src.ingestion.fbref_scraper import FBRefScraper
from src.ingestion.odds_scraper import OddsScraper

logger = structlog.get_logger(__name__)

def run_ingestion_job():
    logger.info("Executing Ingestion Job")
    try:
        # FBRef
        fbref = FBRefScraper()
        fbref.fetch_with_retry(cache_key="fbref_scheduled")
        # Odds
        odds = OddsScraper()
        odds.fetch_with_retry(cache_key="odds_scheduled")
        logger.info("Ingestion completed")
    except Exception as e:
        logger.error("Error in ingestion job", error=str(e))
        AlertManager().send_alert(
            title="Fallo en Ingesta",
            message=f"Error durante la ingesta de datos: {str(e)}",
            level="critical"
        )

def run_retrain_job():
    logger.info("Executing Retraining Job")
    try:
        pipeline = PredictionPipeline()
        metrics = pipeline.train(optimize_hyperparams=False)
        
        brier = metrics.get('brier_score', 1.0)
        if brier > 0.23:
            AlertManager().send_alert(
                title="Degradación de Modelo",
                message=f"Brier score crítico tras reentrenamiento: {brier}",
                level="critical"
            )
        else:
            AlertManager().send_alert(
                title="Reentrenamiento Exitoso",
                message=f"Nuevo Brier Score: {brier}",
                level="info"
            )
    except Exception as e:
        logger.error("Error en retrain job", error=str(e))
        AlertManager().send_alert(
            title="Fallo en Reentrenamiento",
            message=f"Error: {str(e)}",
            level="critical"
        )

def run_simulation_job():
    logger.info("Executing Simulation Job")
    try:
        pipeline = PredictionPipeline()
        # requires team_lambdas, ideally fetched from DB. Placeholder implementation:
        logger.info("Simulation placeholder execution completed")
    except Exception as e:
        logger.error("Error en simulation job", error=str(e))

def start_scheduler():
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
