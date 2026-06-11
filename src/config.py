"""
Configuración global del sistema.
Utiliza pydantic-settings para la carga de variables de entorno.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Conexión DB — componentes individuales (Docker / .env)
    db_host: str = 'localhost'
    db_port: int = 1433
    db_name: str = 'mundial'
    db_user: str = 'sa'
    db_sa_password: str = ''
    db_driver: str = 'ODBC Driver 18 for SQL Server'

    # Conexión DB — override directo (prioridad sobre componentes)
    db_connection_string: str = ''

    # MLflow
    mlflow_tracking_uri: str = 'http://mlflow:5000'

    # Odds API
    odds_api_key: str = ''
    odds_api_base_url: str = 'https://api.the-odds-api.com/v4'

    # Datos
    data_start_year: int = 2014
    cache_dir: str = './data/cache'

    # Modelo XGBoost
    n_simulations: int = 10_000
    xgb_tree_method: str = 'hist'  # CPU only — sin dependencias GPU

    # Optuna — Optimización de hiperparámetros
    optuna_n_trials: int = 100
    optuna_timeout_seconds: int = 3600  # 1 hora máximo

    # Poisson Bivariado — Dixon-Coles
    poisson_rho_default: float = -0.13  # Correlación típica Dixon-Coles
    max_goals_matrix: int = 8  # Matriz de marcadores 0..8 x 0..8

    # Monte Carlo
    mc_random_seed: int = 42  # Semilla para reproducibilidad

    # Bracket config
    bracket_config_path: str = './config/bracket_2026.yaml'

    @property
    def effective_connection_string(self) -> str:
        """Construye el connection string efectivo.

        Prioridad:
        1. ``db_connection_string`` si fue definido explícitamente.
        2. Construido desde componentes individuales si ``db_sa_password`` existe.
        3. SQLite en memoria como fallback seguro para pruebas.
        """
        if self.db_connection_string:
            return self.db_connection_string
        if self.db_sa_password:
            driver = self.db_driver.replace(' ', '+')
            return (
                f"mssql+pyodbc://{self.db_user}:{self.db_sa_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
                f"?driver={driver}&TrustServerCertificate=yes"
            )
        return 'sqlite:///:memory:'

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )


@lru_cache
def get_settings() -> Settings:
    """Retorna la instancia singleton de la configuración."""
    return Settings()
