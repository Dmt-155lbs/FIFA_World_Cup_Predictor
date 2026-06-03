"""
Configuración global del sistema.
Utiliza pydantic-settings para la carga de variables de entorno.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Conexión DB
    db_connection_string: str = 'sqlite:///:memory:' # Valor por defecto seguro para pruebas
    
    # MLflow
    mlflow_tracking_uri: str = 'http://mlflow:5000'
    
    # Odds API
    odds_api_key: str = ''
    odds_api_base_url: str = 'https://api.the-odds-api.com/v4'
    
    # Datos
    data_start_year: int = 2014
    cache_dir: str = './data/cache'
    
    # Modelo
    n_simulations: int = 10_000
    xgb_tree_method: str = 'hist'  # CPU only
    
    # Bracket config
    bracket_config_path: str = './config/bracket_2026.yaml'
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


@lru_cache
def get_settings() -> Settings:
    """Retorna la instancia singleton de la configuración."""
    return Settings()
