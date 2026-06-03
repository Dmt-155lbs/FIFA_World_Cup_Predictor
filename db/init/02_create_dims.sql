-- ============================================================================
-- Archivo : 02_create_dims.sql
-- Propósito: Crear las tablas dimensionales (DIM_) del esquema estrella para
--            el sistema de predicción del Mundial FIFA 2026.
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
-- TABLA: mundial.DIM_TEAM
-- Descripción: Dimensión de selecciones nacionales. Almacena la información
--              básica de cada equipo participante (nombre, código FIFA,
--              confederación y ranking).
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'DIM_TEAM'
)
BEGIN
    CREATE TABLE [mundial].[DIM_TEAM]
    (
        -- Clave primaria auto-incremental
        [team_id]         INT           IDENTITY(1,1)   NOT NULL,

        -- Nombre completo de la selección (ej. "Argentina", "Germany")
        [team_name]       VARCHAR(100)  NOT NULL  DEFAULT 'UNKNOWN',

        -- Código FIFA de tres letras (ej. "ARG", "GER")
        [fifa_code]       CHAR(3)       NOT NULL  DEFAULT 'UNK',

        -- Confederación continental (ej. "CONMEBOL", "UEFA")
        [confederation]   VARCHAR(20)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Posición en el ranking FIFA; -1 indica dato no disponible
        [fifa_ranking]    INT           NOT NULL  DEFAULT -1,

        -- Fecha y hora de última actualización del registro (UTC)
        [updated_at]      DATETIME2     NOT NULL  DEFAULT SYSUTCDATETIME(),

        -- Restricciones
        CONSTRAINT [PK_DIM_TEAM] PRIMARY KEY CLUSTERED ([team_id])
    );

    PRINT '✔ Tabla [mundial].[DIM_TEAM] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[DIM_TEAM] ya existe.';
END
GO

-- ============================================================================
-- ÍNDICE ÚNICO: UQ_TEAM_FIFA_CODE
-- Descripción: Garantiza que no existan dos selecciones con el mismo código
--              FIFA. Esto permite usar fifa_code como clave natural para
--              búsquedas y merges durante la ingesta de datos.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UQ_TEAM_FIFA_CODE'
      AND object_id = OBJECT_ID(N'mundial.DIM_TEAM')
)
BEGIN
    CREATE UNIQUE NONCLUSTERED INDEX [UQ_TEAM_FIFA_CODE]
        ON [mundial].[DIM_TEAM] ([fifa_code]);

    PRINT '✔ Índice único [UQ_TEAM_FIFA_CODE] creado exitosamente.';
END
GO

-- ============================================================================
-- TABLA: mundial.DIM_COMPETITION
-- Descripción: Dimensión de competiciones y torneos. Permite clasificar cada
--              partido por competición, temporada y fase (fase de grupos,
--              octavos de final, etc.).
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND t.name = N'DIM_COMPETITION'
)
BEGIN
    CREATE TABLE [mundial].[DIM_COMPETITION]
    (
        -- Clave primaria auto-incremental
        [competition_id]    INT           IDENTITY(1,1)   NOT NULL,

        -- Nombre de la competición (ej. "FIFA World Cup", "UEFA Nations League")
        [competition_name]  VARCHAR(100)  NOT NULL  DEFAULT 'UNKNOWN',

        -- Temporada o edición (ej. "2026", "2025-2026")
        [season]            VARCHAR(20)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Fase del torneo (ej. "Fase de Grupos", "Octavos de Final")
        [stage]             VARCHAR(50)   NOT NULL  DEFAULT 'UNKNOWN',

        -- Restricciones
        CONSTRAINT [PK_DIM_COMPETITION] PRIMARY KEY CLUSTERED ([competition_id])
    );

    PRINT '✔ Tabla [mundial].[DIM_COMPETITION] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La tabla [mundial].[DIM_COMPETITION] ya existe.';
END
GO

-- ============================================================================
-- ÍNDICE ÚNICO: UQ_COMPETITION_NAME_SEASON
-- Descripción: Garantiza unicidad de la combinación (competition_name, season)
--              para evitar duplicados en la carga de datos de competiciones.
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UQ_COMPETITION_NAME_SEASON'
      AND object_id = OBJECT_ID(N'mundial.DIM_COMPETITION')
)
BEGIN
    CREATE UNIQUE NONCLUSTERED INDEX [UQ_COMPETITION_NAME_SEASON]
        ON [mundial].[DIM_COMPETITION] ([competition_name], [season]);

    PRINT '✔ Índice único [UQ_COMPETITION_NAME_SEASON] creado exitosamente.';
END
GO

PRINT '============================================================';
PRINT ' 02_create_dims.sql ejecutado correctamente.';
PRINT '============================================================';
GO
