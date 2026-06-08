-- ============================================================================
-- Archivo : 03_create_facts.sql
-- Propósito: Crear las tablas de hechos (FACT_) y de seguimiento de ML (ML_)
--            del esquema estrella para el sistema de predicción del Mundial
--            FIFA 2026.
--            Todas las columnas son NOT NULL (Regla de Oro de Persistencia).
--            Los datos faltantes se manejan con valores centinela:
--              -1 (enteros), 0.0 (flotantes), 'UNKNOWN' (cadenas),
--              '1900-01-01' (fechas).
-- Autor    : Sistema de Predicción – Mundial FIFA 2026
-- Fecha    : 2026-06-02
-- ============================================================================

USE [mundial];
GO

-- ============================================================================
-- TABLA: mundial.FACT_MATCH
-- Descripción: Tabla central de hechos. Cada fila representa un partido
--              internacional de fútbol con sus resultados, sede y metadatos.
--              Es la tabla pivote a la que apuntan xG, odds y predicciones.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'FACT_MATCH'
)
BEGIN
    CREATE TABLE [mundial].[FACT_MATCH]
    (
        -- Clave primaria auto-incremental
        [match_id]        BIGINT        IDENTITY(1,1)   NOT NULL,

        -- Referencia a la competición (FK → DIM_COMPETITION)
        [competition_id]  INT           NOT NULL,

        -- Fecha del partido; '1900-01-01' indica dato no disponible
        [match_date]      DATE          NOT NULL  DEFAULT '1900-01-01',

        -- Equipo local (FK → DIM_TEAM)
        [home_team_id]    INT           NOT NULL,

        -- Equipo visitante (FK → DIM_TEAM)
        [away_team_id]    INT           NOT NULL,

        -- Goles anotados por el equipo local
        [home_goals]      SMALLINT      NOT NULL  DEFAULT 0,

        -- Goles anotados por el equipo visitante
        [away_goals]      SMALLINT      NOT NULL  DEFAULT 0,

        -- Sede del partido (nombre del estadio o ciudad)
        [venue]           VARCHAR(200)  NOT NULL  DEFAULT 'UNKNOWN',

        -- ¿Se jugó en terreno neutral? (1 = sí, 0 = no)
        [is_neutral]      BIT           NOT NULL  DEFAULT 0,

        -- ¿Es partido de eliminación directa? (1 = sí, 0 = no)
        [is_knockout]     BIT           NOT NULL  DEFAULT 0,

        -- Asistencia al estadio; -1 indica dato no disponible
        [attendance]      INT           NOT NULL  DEFAULT -1,

        -- Marca temporal de ingesta al data warehouse (UTC)
        [ingested_at]     DATETIME2     NOT NULL  DEFAULT SYSUTCDATETIME(),

        -- =================================================================
        -- Restricciones de la tabla FACT_MATCH
        -- =================================================================

        -- Clave primaria
        CONSTRAINT [PK_FACT_MATCH]
            PRIMARY KEY CLUSTERED ([match_id]),

        -- FK hacia la dimensión de competición
        CONSTRAINT [FK_MATCH_COMP]
            FOREIGN KEY ([competition_id])
            REFERENCES [mundial].[DIM_COMPETITION] ([competition_id]),

        -- FK hacia la dimensión de equipo (local)
        CONSTRAINT [FK_MATCH_HOME]
            FOREIGN KEY ([home_team_id])
            REFERENCES [mundial].[DIM_TEAM] ([team_id]),

        -- FK hacia la dimensión de equipo (visitante)
        CONSTRAINT [FK_MATCH_AWAY]
            FOREIGN KEY ([away_team_id])
            REFERENCES [mundial].[DIM_TEAM] ([team_id]),

        -- Los goles no pueden ser negativos
        CONSTRAINT [CK_MATCH_HOME_GOALS]
            CHECK ([home_goals] >= 0),

        CONSTRAINT [CK_MATCH_AWAY_GOALS]
            CHECK ([away_goals] >= 0)
    );

    PRINT '✔ Tabla [mundial].[FACT_MATCH] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[FACT_MATCH] ya existe.';
END
GO

-- ============================================================================
-- ÍNDICE ÚNICO: UQ_MATCH_IDEMPOTENT
-- Descripción: Garantiza idempotencia en la carga de partidos. La combinación
--              de fecha + equipo local + equipo visitante debe ser única.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UQ_MATCH_IDEMPOTENT'
      AND object_id = OBJECT_ID(N'mundial.FACT_MATCH')
)
BEGIN
    CREATE UNIQUE NONCLUSTERED INDEX [UQ_MATCH_IDEMPOTENT]
        ON [mundial].[FACT_MATCH] ([match_date], [home_team_id], [away_team_id]);

    PRINT '✔ Índice único [UQ_MATCH_IDEMPOTENT] creado exitosamente.';
END
GO

-- ============================================================================
-- TABLA: mundial.FACT_ELO_HISTORY
-- Descripción: Historial de ratings Elo de cada selección a lo largo del
--              tiempo. Permite calcular tendencias y deltas para el modelo
--              predictivo.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'FACT_ELO_HISTORY'
)
BEGIN
    CREATE TABLE [mundial].[FACT_ELO_HISTORY]
    (
        -- Clave primaria auto-incremental
        [elo_id]          BIGINT        IDENTITY(1,1)   NOT NULL,

        -- Referencia al equipo (FK → DIM_TEAM)
        [team_id]         INT           NOT NULL,

        -- Fecha de la medición del rating; '1900-01-01' si no disponible
        [rating_date]     DATE          NOT NULL  DEFAULT '1900-01-01',

        -- Valor del rating Elo en la fecha indicada
        [elo_rating]      FLOAT         NOT NULL  DEFAULT 0.0,

        -- Cambio en el rating Elo respecto a la medición anterior
        [elo_delta]       FLOAT         NOT NULL  DEFAULT 0.0,

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_FACT_ELO_HISTORY]
            PRIMARY KEY CLUSTERED ([elo_id]),

        CONSTRAINT [FK_ELO_TEAM]
            FOREIGN KEY ([team_id])
            REFERENCES [mundial].[DIM_TEAM] ([team_id])
    );

    PRINT '✔ Tabla [mundial].[FACT_ELO_HISTORY] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[FACT_ELO_HISTORY] ya existe.';
END
GO

-- ============================================================================
-- ÍNDICE: IX_ELO_TEAM_DATE
-- Descripción: Índice compuesto para búsquedas rápidas de Elo por equipo
--              y fecha. Esencial para los OUTER APPLY del feature store.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_ELO_TEAM_DATE'
      AND object_id = OBJECT_ID(N'mundial.FACT_ELO_HISTORY')
)
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ELO_TEAM_DATE]
        ON [mundial].[FACT_ELO_HISTORY] ([team_id], [rating_date]);

    PRINT '✔ Índice [IX_ELO_TEAM_DATE] creado exitosamente.';
END
GO

-- ============================================================================
-- TABLA: mundial.FACT_FIFA_RATING
-- Descripción: Valor de mercado y estadísticas agregadas de la plantilla
--              de cada selección. Datos obtenidos de Transfermarkt u otras
--              fuentes de valuación.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'FACT_FIFA_RATING'
)
BEGIN
    CREATE TABLE [mundial].[FACT_FIFA_RATING]
    (
        -- Clave primaria auto-incremental
        [rating_id]              BIGINT        IDENTITY(1,1)   NOT NULL,

        -- Referencia al equipo (FK → DIM_TEAM)
        [team_id]               INT           NOT NULL,

        -- Fecha de la valuación; '1900-01-01' si no disponible
        [valuation_date]        DATE          NOT NULL  DEFAULT '1900-01-01',

        -- Valor de mercado total en EUR
        [overall_rating]        FLOAT         NOT NULL  DEFAULT 0.0,

        -- Tamaño de la plantilla; -1 si no disponible
        [attack_rating]         FLOAT         NOT NULL  DEFAULT 0.0,

        -- Edad promedio de los jugadores
        [midfield_rating]       FLOAT         NOT NULL  DEFAULT 0.0,

        -- Total de internacionalidades acumuladas
        [defence_rating]        FLOAT         NOT NULL  DEFAULT 0.0,

        -- Minutos totales jugados en la temporada de clubes
        

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_FACT_FIFA_RATING]
            PRIMARY KEY CLUSTERED ([rating_id]),

        CONSTRAINT [FK_FIFA_RATING_TEAM]
            FOREIGN KEY ([team_id])
            REFERENCES [mundial].[DIM_TEAM] ([team_id])
    );

    PRINT '✔ Tabla [mundial].[FACT_FIFA_RATING] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[FACT_FIFA_RATING] ya existe.';
END
GO

-- ============================================================================
-- TABLA: mundial.FACT_MATCH_XG
-- Descripción: Estadísticas de goles esperados (xG) asociadas a cada partido.
--              Incluye xG, xGA (goles esperados en contra) y npxG (goles
--              esperados sin penales). Relación 1:1 con FACT_MATCH.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'FACT_MATCH_XG'
)
BEGIN
    CREATE TABLE [mundial].[FACT_MATCH_XG]
    (
        -- Clave primaria auto-incremental
        [xg_id]       BIGINT  IDENTITY(1,1)   NOT NULL,

        -- Referencia al partido (FK → FACT_MATCH)
        [match_id]    BIGINT  NOT NULL,

        -- xG del equipo local
        [home_xg]     FLOAT   NOT NULL  DEFAULT 0.0,

        -- xG del equipo visitante
        [away_xg]     FLOAT   NOT NULL  DEFAULT 0.0,

        -- xGA del equipo local (goles esperados en contra)
        [home_xga]    FLOAT   NOT NULL  DEFAULT 0.0,

        -- xGA del equipo visitante
        [away_xga]    FLOAT   NOT NULL  DEFAULT 0.0,

        -- npxG del equipo local (sin penales)
        [home_npxg]   FLOAT   NOT NULL  DEFAULT 0.0,

        -- npxG del equipo visitante (sin penales)
        [away_npxg]   FLOAT   NOT NULL  DEFAULT 0.0,

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_FACT_MATCH_XG]
            PRIMARY KEY CLUSTERED ([xg_id]),

        CONSTRAINT [FK_XG_MATCH]
            FOREIGN KEY ([match_id])
            REFERENCES [mundial].[FACT_MATCH] ([match_id])
    );

    PRINT '✔ Tabla [mundial].[FACT_MATCH_XG] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[FACT_MATCH_XG] ya existe.';
END
GO

-- ============================================================================
-- TABLA: mundial.FACT_ODDS
-- Descripción: Cuotas de apuestas deportivas recopiladas de distintas casas
--              (bookmakers). Incluye cuotas 1X2 y over/under 2.5 goles.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'FACT_ODDS'
)
BEGIN
    CREATE TABLE [mundial].[FACT_ODDS]
    (
        -- Clave primaria auto-incremental
        [odds_id]       BIGINT        IDENTITY(1,1)   NOT NULL,

        -- Referencia al partido (FK → FACT_MATCH)
        [match_id]      BIGINT        NOT NULL,

        -- Nombre de la casa de apuestas; 'UNKNOWN' si no disponible
        [bookmaker]     VARCHAR(50)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Cuota para victoria local
        [odds_home]     FLOAT         NOT NULL  DEFAULT 0.0,

        -- Cuota para empate
        [odds_draw]     FLOAT         NOT NULL  DEFAULT 0.0,

        -- Cuota para victoria visitante
        [odds_away]     FLOAT         NOT NULL  DEFAULT 0.0,

        -- Cuota para más de 2.5 goles
        [odds_over25]   FLOAT         NOT NULL  DEFAULT 0.0,

        -- Fecha y hora de recopilación de las cuotas (UTC)
        [scraped_at]    DATETIME2     NOT NULL  DEFAULT SYSUTCDATETIME(),

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_FACT_ODDS]
            PRIMARY KEY CLUSTERED ([odds_id]),

        CONSTRAINT [FK_ODDS_MATCH]
            FOREIGN KEY ([match_id])
            REFERENCES [mundial].[FACT_MATCH] ([match_id])
    );

    PRINT '✔ Tabla [mundial].[FACT_ODDS] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[FACT_ODDS] ya existe.';
END
GO

-- ============================================================================
-- TABLA: mundial.FACT_PREDICTIONS
-- Descripción: Predicciones generadas por el modelo de ML para cada partido
--              y equipo. Incluye probabilidades (win/draw/lose), goles
--              esperados y fortalezas ofensiva/defensiva estimadas.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'FACT_PREDICTIONS'
)
BEGIN
    CREATE TABLE [mundial].[FACT_PREDICTIONS]
    (
        -- Clave primaria auto-incremental
        [prediction_id]       BIGINT        IDENTITY(1,1)   NOT NULL,

        -- Referencia al partido (FK → FACT_MATCH)
        [match_id]            BIGINT        NOT NULL,

        -- Equipo sobre el cual se realiza la predicción (FK → DIM_TEAM)
        [team_id]             INT           NOT NULL,

        -- Versión del modelo que generó la predicción
        [model_version]       VARCHAR(50)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Probabilidad de victoria
        [prob_win]            FLOAT         NOT NULL  DEFAULT 0.0,

        -- Probabilidad de empate
        [prob_draw]           FLOAT         NOT NULL  DEFAULT 0.0,

        -- Probabilidad de derrota
        [prob_lose]           FLOAT         NOT NULL  DEFAULT 0.0,

        -- Goles esperados predichos por el modelo
        [expected_goals]      FLOAT         NOT NULL  DEFAULT 0.0,

        -- Fortaleza ofensiva estimada
        [strength_offensive]  FLOAT         NOT NULL  DEFAULT 0.0,

        -- Fortaleza defensiva estimada
        [strength_defensive]  FLOAT         NOT NULL  DEFAULT 0.0,

        -- Fecha y hora de generación de la predicción (UTC)
        [predicted_at]        DATETIME2     NOT NULL  DEFAULT SYSUTCDATETIME(),

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_FACT_PREDICTIONS]
            PRIMARY KEY CLUSTERED ([prediction_id]),

        CONSTRAINT [FK_PRED_MATCH]
            FOREIGN KEY ([match_id])
            REFERENCES [mundial].[FACT_MATCH] ([match_id]),

        CONSTRAINT [FK_PRED_TEAM]
            FOREIGN KEY ([team_id])
            REFERENCES [mundial].[DIM_TEAM] ([team_id])
    );

    PRINT '✔ Tabla [mundial].[FACT_PREDICTIONS] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[FACT_PREDICTIONS] ya existe.';
END
GO

-- ============================================================================
-- TABLA: mundial.ML_EXPERIMENT
-- Descripción: Registro de experimentos de Machine Learning. Cada fila
--              representa un experimento (conjunto de hiperparámetros y
--              configuración) con su referencia a MLflow.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'ML_EXPERIMENT'
)
BEGIN
    CREATE TABLE [mundial].[ML_EXPERIMENT]
    (
        -- Clave primaria auto-incremental
        [experiment_id]    INT           IDENTITY(1,1)   NOT NULL,

        -- Nombre descriptivo del experimento
        [experiment_name]  VARCHAR(200)  NOT NULL  DEFAULT 'UNKNOWN',

        -- ID del run en MLflow para trazabilidad
        [mlflow_run_id]    VARCHAR(64)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Fecha y hora de creación del experimento (UTC)
        [created_at]       DATETIME2     NOT NULL  DEFAULT SYSUTCDATETIME(),

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_ML_EXPERIMENT]
            PRIMARY KEY CLUSTERED ([experiment_id])
    );

    PRINT '✔ Tabla [mundial].[ML_EXPERIMENT] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[ML_EXPERIMENT] ya existe.';
END
GO

-- ============================================================================
-- TABLA: mundial.ML_MODEL_VERSION
-- Descripción: Versiones de modelos entrenados, asociadas a un experimento.
--              Almacena métricas de evaluación (Brier score, ROI del backtest,
--              log-loss) y la ruta al artefacto serializado del modelo.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'ML_MODEL_VERSION'
)
BEGIN
    CREATE TABLE [mundial].[ML_MODEL_VERSION]
    (
        -- Clave primaria auto-incremental
        [model_version_id]  INT           IDENTITY(1,1)   NOT NULL,

        -- Referencia al experimento padre (FK → ML_EXPERIMENT)
        [experiment_id]     INT           NOT NULL,

        -- Etiqueta de versión (ej. "v1.2.0", "baseline_xgb")
        [version_tag]       VARCHAR(50)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Brier score del modelo (métrica de calibración; menor = mejor)
        [brier_score]       FLOAT         NOT NULL  DEFAULT 0.0,

        -- ROI del backtest de apuestas (retorno sobre inversión simulado)
        [roi_backtest]      FLOAT         NOT NULL  DEFAULT 0.0,

        -- Log-loss del modelo (entropía cruzada; menor = mejor)
        [log_loss]          FLOAT         NOT NULL  DEFAULT 0.0,

        -- Ruta al artefacto serializado del modelo (pickle, ONNX, etc.)
        [artifact_path]     VARCHAR(500)  NOT NULL  DEFAULT 'UNKNOWN',

        -- Fecha y hora de entrenamiento del modelo (UTC)
        [trained_at]        DATETIME2     NOT NULL  DEFAULT SYSUTCDATETIME(),

        -- =================================================================
        -- Restricciones
        -- =================================================================
        CONSTRAINT [PK_ML_MODEL_VERSION]
            PRIMARY KEY CLUSTERED ([model_version_id]),

        CONSTRAINT [FK_MODEL_EXPERIMENT]
            FOREIGN KEY ([experiment_id])
            REFERENCES [mundial].[ML_EXPERIMENT] ([experiment_id])
    );

    PRINT '✔ Tabla [mundial].[ML_MODEL_VERSION] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[ML_MODEL_VERSION] ya existe.';
END
GO

PRINT '============================================================';
PRINT ' 03_create_facts.sql ejecutado correctamente.';
PRINT ' Tablas creadas:';
PRINT '   - mundial.FACT_MATCH';
PRINT '   - mundial.FACT_ELO_HISTORY';
PRINT '   - mundial.FACT_FIFA_RATING';
PRINT '   - mundial.FACT_MATCH_XG';
PRINT '   - mundial.FACT_ODDS';
PRINT '   - mundial.FACT_PREDICTIONS';
PRINT '   - mundial.ML_EXPERIMENT';
PRINT '   - mundial.ML_MODEL_VERSION';
PRINT '============================================================';
GO
