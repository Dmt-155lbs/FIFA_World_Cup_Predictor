# 🏆 Mundial 2026 — Sistema de Predicción

> **Sistema de predicción probabilística para el Mundial FIFA 2026** basado en un Ensemble Híbrido que combina XGBoost, Modelo de Poisson Bivariado y Simulación Monte Carlo.

---

## 📋 Descripción

Este proyecto implementa un sistema completo de predicción de resultados para la Copa Mundial de la FIFA 2026. El núcleo del sistema es un **Ensemble Híbrido** que integra tres enfoques complementarios:

| Componente | Descripción |
|---|---|
| **XGBoost** | Modelo de gradient boosting entrenado con features históricas, ELO, forma reciente y cuotas de apuestas. Ejecuta en CPU (`tree_method='hist'`). |
| **Poisson Bivariado** | Modelo estadístico que estima la distribución conjunta de goles marcados por cada equipo, capturando la correlación entre ataques y defensas. |
| **Monte Carlo** | Simulación de miles de escenarios del torneo completo (fase de grupos → eliminatorias → final) para estimar probabilidades de avance y campeonato. |

El sistema incluye:
- 📊 **Dashboard interactivo** con Streamlit para explorar predicciones en tiempo real.
- 🧪 **Seguimiento de experimentos** con MLflow para versionar modelos y comparar métricas.
- 🗄️ **Base de datos SQL Server** para almacenar datos históricos, features y predicciones.
- 🔄 **Pipelines automatizados** de ingesta, feature engineering, entrenamiento y predicción.

---

## ⚙️ Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- **8 GB de RAM** disponibles como mínimo (SQL Server requiere al menos 2 GB)

---

## 🚀 Inicio Rápido

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Mundial
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales reales
```

### 3. Levantar los servicios

```bash
docker-compose up -d
```

### 4. Verificar que todo funciona

```bash
docker-compose ps
```

### 5. Acceder a las interfaces

| Servicio | URL | Descripción |
|---|---|---|
| **Streamlit** | [http://localhost:8501](http://localhost:8501) | Dashboard de predicciones |
| **MLflow** | [http://localhost:5000](http://localhost:5000) | Seguimiento de experimentos |
| **SQL Server** | `localhost:1433` | Base de datos (usar SSMS o Azure Data Studio) |

---

## 📁 Estructura del Proyecto

```
Mundial/
├── docker-compose.yml          # Orquestación de servicios
├── .env.example                # Plantilla de variables de entorno
├── .gitignore                  # Archivos excluidos de Git
├── pyproject.toml              # Configuración del proyecto Python
├── README.md                   # Este archivo
│
├── docker/                     # Dockerfiles por servicio
│   ├── app/
│   │   └── Dockerfile          # Servicio principal (pipelines, modelos)
│   ├── mlflow/
│   │   └── Dockerfile          # Servidor MLflow
│   └── streamlit/
│       └── Dockerfile          # Dashboard Streamlit
│
├── db/                         # Scripts de base de datos
│   └── init/                   # Scripts DDL de inicialización
│
├── src/                        # Código fuente principal
│   ├── config/                 # Configuración y settings
│   ├── data/                   # Ingesta y procesamiento de datos
│   ├── features/               # Feature engineering
│   ├── models/                 # Modelos de predicción
│   ├── simulation/             # Simulación Monte Carlo
│   ├── evaluation/             # Métricas y calibración
│   ├── dashboard/              # Aplicación Streamlit
│   └── utils/                  # Utilidades compartidas
│
├── data/                       # Datos (no versionados)
│   ├── raw/                    # Datos crudos descargados
│   ├── processed/              # Datos procesados
│   └── cache/                  # Caché temporal
│
├── tests/                      # Tests unitarios e integración
│
├── notebooks/                  # Notebooks exploratorios
│
└── configs/                    # Archivos de configuración YAML
```

---

## 🛠️ Stack Tecnológico

| Categoría | Tecnología | Versión |
|---|---|---|
| **Lenguaje** | Python | 3.11+ |
| **ML / Predicción** | XGBoost, scikit-learn, SciPy | 2.1+, latest, latest |
| **Optimización** | Optuna | latest |
| **Interpretabilidad** | SHAP | 0.45+ |
| **Simulación** | NumPy, Numba | latest |
| **Datos Fútbol** | soccerdata | 1.3+ |
| **Base de Datos** | SQL Server 2022 | Developer |
| **ORM** | SQLAlchemy | 2.0+ |
| **Experimentos** | MLflow | 2.14+ |
| **Dashboard** | Streamlit, Plotly | 1.36+, latest |
| **Validación** | Pydantic | 2.0+ |
| **Contenedores** | Docker, Docker Compose | 20.10+, 2.0+ |

---

## 🧪 Ejecutar Tests

```bash
# Desde dentro del contenedor 'app'
docker-compose exec app pytest

# Con cobertura
docker-compose exec app pytest --cov=src --cov-report=html
```

---

## 📝 Comandos Útiles

```bash
# Ver logs de un servicio específico
docker-compose logs -f app

# Reiniciar un servicio
docker-compose restart mlflow

# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes (⚠️ borra datos)
docker-compose down -v

# Reconstruir imágenes después de cambios en Dockerfiles
docker-compose build --no-cache
```

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  Desarrollado con ⚽ para el Mundial FIFA 2026
</p>
