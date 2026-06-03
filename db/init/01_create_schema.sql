-- ============================================================================
-- Archivo : 01_create_schema.sql
-- Propósito: Crear la base de datos "mundial" y el schema "mundial" dentro
--            de ella. Este script es idempotente: verifica la existencia
--            antes de crear cada objeto.
-- Autor    : Sistema de Predicción – Mundial FIFA 2026
-- Fecha    : 2026-06-02
-- ============================================================================

-- ============================================================================
-- PASO 1: Crear la base de datos si no existe
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = N'mundial')
BEGIN
    CREATE DATABASE [mundial]
        COLLATE Latin1_General_CI_AS;
    PRINT '✔ Base de datos [mundial] creada exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ La base de datos [mundial] ya existe. No se realizaron cambios.';
END
GO

-- ============================================================================
-- PASO 2: Cambiar al contexto de la base de datos "mundial"
-- ============================================================================
USE [mundial];
GO

-- ============================================================================
-- PASO 3: Crear el schema "mundial" dentro de la base de datos
-- Se usa este schema para aislar todas las tablas del proyecto de predicción
-- del Mundial FIFA 2026, evitando colisiones con otros objetos en dbo.
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'mundial')
BEGIN
    EXEC('CREATE SCHEMA [mundial] AUTHORIZATION [dbo]');
    PRINT '✔ Schema [mundial] creado exitosamente.';
END
ELSE
BEGIN
    PRINT 'ℹ El schema [mundial] ya existe. No se realizaron cambios.';
END
GO

PRINT '============================================================';
PRINT ' 01_create_schema.sql ejecutado correctamente.';
PRINT '============================================================';
GO
