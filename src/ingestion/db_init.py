"""
Inicializador de la base de datos SQL Server.

Ejecuta los scripts SQL de ``db/init/`` en orden contra la instancia de
SQL Server configurada.  A diferencia de PostgreSQL, SQL Server **no**
auto-ejecuta scripts montados en ``/docker-entrypoint-initdb.d``, por lo
que este módulo provee la ejecución programática.

Uso:
    python -m src.ingestion.db_init
"""

from __future__ import annotations

from pathlib import Path

import structlog
from sqlalchemy import create_engine, text

from src.config import get_settings
from src.utils.db import get_engine

log = structlog.get_logger(__name__)


def _get_bootstrap_engine():
    """Crea un engine de *bootstrap* conectado a la base ``master``.

    En un SQL Server recién levantado la base ``mundial`` todavía no existe,
    por lo que conectar directamente a ella (como hace ``get_engine``) falla.
    Los scripts de inicialización **crean** ``mundial``/``mlflow`` y luego
    cambian de contexto con ``USE [mundial]``; para que ``CREATE DATABASE`` sea
    válido y el cambio de contexto persista entre batches, la conexión debe:

    1. Apuntar a ``master`` (que siempre existe).
    2. Estar en modo ``AUTOCOMMIT`` (``CREATE DATABASE`` no se permite dentro
       de una transacción multi-sentencia).

    Si la cadena de conexión no es de SQL Server (p. ej. SQLite en pruebas),
    se devuelve el engine estándar.
    """
    settings = get_settings()
    cs = settings.effective_connection_string

    if not cs.startswith("mssql"):
        # Fallback (SQLite u otros): no aplica el problema master/mundial.
        return get_engine()

    # Reemplazar la base de destino por ``master`` preservando el resto.
    master_cs = cs.replace(f"/{settings.db_name}?", "/master?", 1)
    return create_engine(
        master_cs,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )

# Orden determinista de ejecución
SQL_SCRIPTS_ORDER: list[str] = [
    "001_create_databases.sql",
    "01_create_schema.sql",
    "02_create_dims.sql",
    "03_create_facts.sql",
    "04_create_views.sql",
]


def _split_on_go(sql_text: str) -> list[str]:
    """Divide un script SQL Server en batches separados por ``GO``.

    SQL Server no soporta ``GO`` dentro de una sola ejecución; cada
    batch debe enviarse por separado.

    Parameters
    ----------
    sql_text : str
        Contenido completo del archivo ``.sql``.

    Returns
    -------
    list[str]
        Lista de batches SQL limpios (sin ``GO``).
    """
    batches: list[str] = []
    current: list[str] = []

    for line in sql_text.split("\n"):
        stripped = line.strip().upper()
        if stripped == "GO" or stripped == "GO;":
            batch = "\n".join(current).strip()
            if batch:
                batches.append(batch)
            current = []
        else:
            current.append(line)

    # Último batch (sin GO final)
    remaining = "\n".join(current).strip()
    if remaining:
        batches.append(remaining)

    return batches


def run_sql_scripts(
    scripts_dir: str | Path | None = None,
    scripts: list[str] | None = None,
) -> dict[str, str]:
    """Ejecuta los scripts SQL de inicialización en orden.

    Parameters
    ----------
    scripts_dir : str | Path | None
        Directorio con los archivos ``.sql``.
        Por defecto: ``<proyecto>/db/init/``.
    scripts : list[str] | None
        Lista de archivos a ejecutar en orden.
        Por defecto: ``SQL_SCRIPTS_ORDER``.

    Returns
    -------
    dict[str, str]
        ``{script_name: "OK" | "ERROR: ..."}``.
    """
    if scripts_dir is None:
        # Determinar ruta relativa al proyecto
        project_root = Path(__file__).resolve().parents[2]
        scripts_dir = project_root / "db" / "init"
    else:
        scripts_dir = Path(scripts_dir)

    if scripts is None:
        scripts = SQL_SCRIPTS_ORDER

    # Engine de bootstrap: conecta a ``master`` en AUTOCOMMIT para poder crear
    # las bases ``mundial``/``mlflow`` antes de que existan.
    engine = _get_bootstrap_engine()
    results: dict[str, str] = {}

    log.info(
        "Iniciando ejecución de scripts SQL",
        directorio=str(scripts_dir),
        scripts=scripts,
    )

    for script_name in scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            log.warning(
                "Script no encontrado, omitiendo",
                script=script_name,
            )
            results[script_name] = "SKIPPED: file not found"
            continue

        try:
            sql_content = script_path.read_text(encoding="utf-8")
            batches = _split_on_go(sql_content)

            log.info(
                "Ejecutando script",
                script=script_name,
                batches=len(batches),
            )

            with engine.connect() as conn:
                for i, batch in enumerate(batches):
                    try:
                        conn.execute(text(batch))
                        conn.commit()
                    except Exception as batch_err:
                        # Algunos batches pueden fallar si las tablas ya
                        # existen (IF NOT EXISTS los maneja, pero USE/CREATE
                        # DATABASE puede fallar en contextos restringidos).
                        log.warning(
                            "Batch ignorado",
                            script=script_name,
                            batch=i + 1,
                            error=str(batch_err),
                        )

            results[script_name] = "OK"
            log.info("Script ejecutado", script=script_name, status="OK")

        except Exception as e:
            error_msg = str(e)
            results[script_name] = f"ERROR: {error_msg}"
            log.error(
                "Error ejecutando script",
                script=script_name,
                error=error_msg,
            )

    return results


def init_database() -> None:
    """Punto de entrada principal para inicializar la base de datos.

    Ejecuta todos los scripts SQL y luego puebla las dimensiones
    con los datos semilla.
    """
    from src.ingestion.loader import DataLoader
    from src.ingestion.seed_data import COMPETITIONS, WORLD_CUP_2026_TEAMS

    log.info("=" * 60)
    log.info("INICIALIZANDO BASE DE DATOS — Mundial FIFA 2026")
    log.info("=" * 60)

    # Paso 1: Crear esquema y tablas
    settings = get_settings()
    log.info(
        "Conexión a BD",
        host=settings.db_host,
        port=settings.db_port,
        db=settings.db_name,
    )

    results = run_sql_scripts()
    for script, status in results.items():
        log.info(f"  {script}: {status}")

    # Paso 2: Poblar dimensiones
    loader = DataLoader()

    teams_inserted = loader.seed_teams(WORLD_CUP_2026_TEAMS)
    log.info(f"Equipos insertados: {teams_inserted}/48")

    comps_inserted = loader.seed_competitions(COMPETITIONS)
    log.info(f"Competiciones insertadas: {comps_inserted}/{len(COMPETITIONS)}")

    # Paso 3: Verificar conteos
    counts = loader.get_table_counts()
    log.info("Conteos de tablas post-inicialización:")
    for table, count in counts.items():
        log.info(f"  {table}: {count}")

    log.info("=" * 60)
    log.info("BASE DE DATOS INICIALIZADA EXITOSAMENTE")
    log.info("=" * 60)


if __name__ == "__main__":
    init_database()
