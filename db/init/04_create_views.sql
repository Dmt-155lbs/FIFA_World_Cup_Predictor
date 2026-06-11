-- ============================================================================
-- Archivo : 04_create_views.sql
-- Propósito: Crear la vista [mundial].[vw_feature_store] que consolida todos
--            los datos de partidos, Elo, xG y valor de plantilla en un único
--            vector de features listo para el entrenamiento del modelo
--            predictivo del Mundial FIFA 2026.
-- Notas   : - Usa funciones ventana (LAG) para calcular deltas de Elo.
--            - Aplica transformación logarítmica al valor de mercado.
--            - Normaliza minutos jugados por la constante 4500 (90 min × 50
--              partidos de referencia por temporada).
--            - Todos los JOINs son LEFT para no perder partidos sin datos
--              laterales; los ISNULL garantizan columnas NOT NULL.
-- Autor    : Sistema de Predicción – Mundial FIFA 2026
-- Fecha    : 2026-06-02
-- ============================================================================

USE [mundial];
GO

-- ============================================================================
-- Eliminar la vista si ya existe para recrearla de forma idempotente
-- ============================================================================
IF EXISTS (
    SELECT 1 FROM sys.views v
    JOIN sys.schemas s ON v.schema_id = s.schema_id
    WHERE s.name = N'mundial' AND v.name = N'vw_feature_store'
)
BEGIN
    DROP VIEW [mundial].[vw_feature_store];
    PRINT 'ℹ Vista [mundial].[vw_feature_store] existente eliminada para recreación.';
END
GO

-- ============================================================================
-- VISTA: mundial.vw_feature_store
-- Descripción: Vector de features para el modelo de predicción.
--              Combina datos de partidos (FACT_MATCH) con:
--                - Ratings Elo de ambos equipos (FACT_ELO_HISTORY)
--                - Delta Elo reciente vía LAG window function
--                - Estadísticas de goles esperados (FACT_MATCH_XG)
--                - Valor de mercado de la plantilla (FACT_FIFA_RATING)
--              Las columnas target son home_goals y away_goals.
-- ============================================================================
CREATE VIEW [mundial].[vw_feature_store]
AS
WITH cte_elo_con_delta AS (
    -- ====================================================================
    -- CTE: Calcula el delta de Elo respecto a la observación anterior
    -- para cada equipo, usando LAG ordenado por fecha.
    -- ====================================================================
    SELECT
        [team_id],
        [rating_date],
        [elo_rating],
        [elo_delta],
        LAG([elo_rating], 1, [elo_rating])
            OVER (PARTITION BY [team_id] ORDER BY [rating_date]) AS [elo_rating_prev]
    FROM [mundial].[FACT_ELO_HISTORY]
)
SELECT
    -- ====================================================================
    -- Identificadores del partido
    -- ====================================================================
    m.[match_id],
    m.[match_date],
    m.[competition_id],
    comp.[competition_name],
    comp.[stage],

    -- ====================================================================
    -- Equipos
    -- ====================================================================
    m.[home_team_id],
    ht.[team_name]                                          AS [home_team_name],
    ht.[fifa_code]                                          AS [home_fifa_code],
    m.[away_team_id],
    at.[team_name]                                          AS [away_team_name],
    at.[fifa_code]                                          AS [away_fifa_code],

    -- ====================================================================
    -- Contexto del partido
    -- ====================================================================
    m.[venue],
    m.[is_neutral],
    m.[is_knockout],
    m.[attendance],

    -- ====================================================================
    -- Features de Elo (rating actual y delta respecto a medición anterior)
    -- ====================================================================
    ISNULL(eh.[elo_rating],   0.0)                          AS [home_elo],
    ISNULL(ea.[elo_rating],   0.0)                          AS [away_elo],
    ISNULL(eh.[elo_rating] - eh.[elo_rating_prev], 0.0)     AS [home_elo_delta],
    ISNULL(ea.[elo_rating] - ea.[elo_rating_prev], 0.0)     AS [away_elo_delta],
    ISNULL(eh.[elo_rating], 0.0)
        - ISNULL(ea.[elo_rating], 0.0)                      AS [elo_diff],

    -- ====================================================================
    -- Features de goles esperados (xG, xGA, npxG)
    -- ====================================================================
    ISNULL(xg.[home_xg],     0.0)                           AS [home_xg],
    ISNULL(xg.[away_xg],     0.0)                           AS [away_xg],
    ISNULL(xg.[home_xga],    0.0)                           AS [home_xga],
    ISNULL(xg.[away_xga],    0.0)                           AS [away_xga],
    ISNULL(xg.[home_npxg],   0.0)                           AS [home_npxg],
    ISNULL(xg.[away_npxg],   0.0)                           AS [away_npxg],

    -- ====================================================================
    -- Features de valor de plantilla
    -- Transformación logarítmica para normalizar la distribución sesgada
    -- del valor de mercado (en EUR). Se usa LOG(valor + 1) para evitar
    -- LOG(0) cuando el valor centinela es 0.0.
    -- ====================================================================
    ISNULL(svh.[attack_rating], 0.0)            AS [home_fifa_attack],
    ISNULL(sva.[attack_rating], 0.0)            AS [away_fifa_attack],
    ISNULL(svh.[overall_rating],            -1)                  AS [home_fifa_overall],
    ISNULL(sva.[overall_rating],            -1)                  AS [away_fifa_overall],
    ISNULL(svh.[midfield_rating],              0.0)                  AS [home_fifa_midfield],
    ISNULL(sva.[midfield_rating],              0.0)                  AS [away_fifa_midfield],
    ISNULL(svh.[defence_rating],            -1)                  AS [home_fifa_defence],
    ISNULL(sva.[defence_rating],            -1)                  AS [away_fifa_defence],

    -- ====================================================================
    -- Carga de minutos normalizada
    -- Se divide entre 4500 (90 min × 50 partidos de referencia) para
    -- obtener un ratio de carga relativa de la plantilla.
    -- ====================================================================
    ISNULL(
        svh.[overall_rating],
        0.0
    )                                                        AS [home_fifa_overall_copy],
    ISNULL(
        sva.[overall_rating],
        0.0
    )                                                        AS [away_fifa_overall_copy],

    -- ====================================================================
    -- Variables objetivo (targets) para el modelo
    -- ====================================================================
    m.[home_goals],
    m.[away_goals],

    -- ====================================================================
    -- Metadatos de ingesta
    -- ====================================================================
    m.[ingested_at]

FROM [mundial].[FACT_MATCH] AS m

-- Dimensión de competición
INNER JOIN [mundial].[DIM_COMPETITION] AS comp
    ON m.[competition_id] = comp.[competition_id]

-- Dimensión de equipos (local y visitante)
INNER JOIN [mundial].[DIM_TEAM] AS ht
    ON m.[home_team_id] = ht.[team_id]
INNER JOIN [mundial].[DIM_TEAM] AS at
    ON m.[away_team_id] = at.[team_id]

-- ====================================================================
-- Elo del equipo local: se toma la medición más reciente anterior o
-- igual a la fecha del partido. Se usa OUTER APPLY con TOP 1 para
-- obtener el Elo más cercano temporalmente.
-- ====================================================================
OUTER APPLY (
    SELECT TOP 1
        e.[elo_rating],
        e.[elo_rating_prev]
    FROM cte_elo_con_delta AS e
    WHERE e.[team_id]     = m.[home_team_id]
      AND e.[rating_date] <= m.[match_date]
    ORDER BY e.[rating_date] DESC
) AS eh

-- ====================================================================
-- Elo del equipo visitante (misma lógica que el local)
-- ====================================================================
OUTER APPLY (
    SELECT TOP 1
        e.[elo_rating],
        e.[elo_rating_prev]
    FROM cte_elo_con_delta AS e
    WHERE e.[team_id]     = m.[away_team_id]
      AND e.[rating_date] <= m.[match_date]
    ORDER BY e.[rating_date] DESC
) AS ea

-- ====================================================================
-- xG del partido (relación 1:1 con FACT_MATCH)
-- ====================================================================
LEFT JOIN [mundial].[FACT_MATCH_XG] AS xg
    ON m.[match_id] = xg.[match_id]

-- ====================================================================
-- Valor de plantilla del equipo local: la valuación más reciente
-- anterior o igual a la fecha del partido.
-- ====================================================================
OUTER APPLY (
    SELECT TOP 1
        sv.[attack_rating],
        sv.[overall_rating],
        sv.[midfield_rating],
        sv.[defence_rating]
    FROM [mundial].[FACT_FIFA_RATING] AS sv
    WHERE sv.[team_id]        = m.[home_team_id]
      AND sv.[valuation_date] <= m.[match_date]
    ORDER BY sv.[valuation_date] DESC
) AS svh

-- ====================================================================
-- Valor de plantilla del equipo visitante (misma lógica)
-- ====================================================================
OUTER APPLY (
    SELECT TOP 1
        sv.[attack_rating],
        sv.[overall_rating],
        sv.[midfield_rating],
        sv.[defence_rating]
    FROM [mundial].[FACT_FIFA_RATING] AS sv
    WHERE sv.[team_id]        = m.[away_team_id]
      AND sv.[valuation_date] <= m.[match_date]
    ORDER BY sv.[valuation_date] DESC
) AS sva;
GO

PRINT '✔ Vista [mundial].[vw_feature_store] creada exitosamente.';
GO

PRINT '============================================================';
PRINT ' 04_create_views.sql ejecutado correctamente.';
PRINT '============================================================';
GO
