-- =============================================================================
-- Script de inicialización — SQL Server 2022
-- =============================================================================
-- Este script se ejecuta al iniciar el contenedor de SQL Server por primera vez.
-- Crea las bases de datos necesarias para el proyecto.
-- =============================================================================

-- Crear base de datos principal del proyecto
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'mundial')
BEGIN
    CREATE DATABASE mundial;
    PRINT 'Base de datos [mundial] creada exitosamente.';
END
GO

-- Crear base de datos para MLflow
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'mlflow')
BEGIN
    CREATE DATABASE mlflow;
    PRINT 'Base de datos [mlflow] creada exitosamente.';
END
GO

USE mundial;
GO

PRINT 'Inicialización completada.';
GO
