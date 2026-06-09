"""
Script standalone para ejecutar únicamente el scraper de The Odds API.
"""
import sys
import structlog
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Asegurar que el path del proyecto esté en sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.ingestion.odds_scraper import OddsScraperLive

logger = structlog.get_logger()

def main():
    logger.info("Iniciando descarga de cuotas de apuestas (The Odds API)...")
    scraper = OddsScraperLive()
    df = scraper.fetch_with_retry()
    
    if df is None or df.empty:
        logger.warning("No se obtuvieron cuotas o hubo un error en la conexión.")
    else:
        logger.info(f"Cuotas de consenso calculadas exitosamente. Partidos obtenidos: {len(df)}")
        print("\n=== MUESTRA DE CUOTAS DE CONSENSO ===")
        print(df.head())
        print("=====================================\n")
        
    logger.info("Proceso de scraping finalizado.")

if __name__ == "__main__":
    main()
