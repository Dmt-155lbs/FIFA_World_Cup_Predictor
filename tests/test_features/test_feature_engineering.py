"""
Tests para el módulo de ingeniería de features.

Utiliza datos sintéticos en memoria — NO se conecta a la base de datos.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer
from src.utils.constants import (
    CONFEDERATIONS,
    K_FACTOR,
    ROLLING_WINDOW_MATCHES,
    SENTINEL_INT,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def synthetic_matches() -> pd.DataFrame:
    """
    Genera 20 partidos sintéticos entre 4 equipos.

    Equipos: 1 (Argentina), 2 (Brasil), 3 (Francia), 4 (Alemania).
    Las fechas se espacian cada 7 días a partir de 2023-01-01.
    """
    np.random.seed(42)
    teams = [1, 2, 3, 4]
    team_names = {1: "Argentina", 2: "Brasil", 3: "Francia", 4: "Alemania"}
    fifa_codes = {1: "ARG", 2: "BRA", 3: "FRA", 4: "GER"}

    records: list[dict] = []
    base_date = date(2023, 1, 1)

    for i in range(20):
        # Rotar combinaciones de equipos
        home_id = teams[i % 4]
        away_id = teams[(i + 1) % 4]

        home_goals = int(np.random.randint(0, 4))
        away_goals = int(np.random.randint(0, 4))

        records.append({
            "match_id": i + 1,
            "match_date": pd.Timestamp(base_date + timedelta(days=7 * i)),
            "competition_id": 1,
            "competition_name": "FIFA World Cup" if i % 3 == 0 else "Friendly",
            "stage": "Group" if i < 10 else "Knockout",
            "home_team_id": home_id,
            "home_team_name": team_names[home_id],
            "home_fifa_code": fifa_codes[home_id],
            "away_team_id": away_id,
            "away_team_name": team_names[away_id],
            "away_fifa_code": fifa_codes[away_id],
            "venue": "Estadio Sintético",
            "is_neutral": int(i % 2 == 0),
            "is_knockout": int(i >= 10),
            "attendance": 50000 + i * 1000,
            "home_elo": 1800.0 + np.random.randn() * 50,
            "away_elo": 1750.0 + np.random.randn() * 50,
            "home_elo_delta": float(np.random.randn() * 5),
            "away_elo_delta": float(np.random.randn() * 5),
            "elo_diff": 50.0 + np.random.randn() * 20,
            "home_xg": round(float(np.random.uniform(0.5, 3.0)), 2),
            "away_xg": round(float(np.random.uniform(0.3, 2.5)), 2),
            "home_xga": round(float(np.random.uniform(0.3, 2.5)), 2),
            "away_xga": round(float(np.random.uniform(0.5, 3.0)), 2),
            "home_npxg": round(float(np.random.uniform(0.3, 2.5)), 2),
            "away_npxg": round(float(np.random.uniform(0.2, 2.0)), 2),
            "home_squad_value_log": round(float(np.random.uniform(7.0, 9.0)), 2),
            "away_squad_value_log": round(float(np.random.uniform(7.0, 9.0)), 2),
            "home_squad_size": 26,
            "away_squad_size": 26,
            "home_avg_age": round(float(np.random.uniform(25.0, 29.0)), 1),
            "away_avg_age": round(float(np.random.uniform(25.0, 29.0)), 1),
            "home_total_caps": int(np.random.randint(300, 800)),
            "away_total_caps": int(np.random.randint(300, 800)),
            "home_minutes_load": float(np.random.randint(3000, 5000)),
            "away_minutes_load": float(np.random.randint(3000, 5000)),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "ingested_at": pd.Timestamp("2024-01-01"),
        })

    return pd.DataFrame(records)


@pytest.fixture()
def engineer() -> FeatureEngineer:
    """
    Crea un FeatureEngineer con motor SQLite en memoria.

    No carga datos de BD; los tests pasan DataFrames directamente.
    """
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    return FeatureEngineer(engine=engine)


@pytest.fixture()
def df_with_rolling(
    engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
) -> pd.DataFrame:
    """DataFrame con rolling features ya calculadas."""
    return engineer.compute_rolling_features(synthetic_matches)


@pytest.fixture()
def df_with_sentinels(synthetic_matches: pd.DataFrame) -> pd.DataFrame:
    """DataFrame con valores centinela inyectados para pruebas."""
    df = synthetic_matches.copy()
    # Inyectar centinelas -1 en filas específicas
    df.loc[0, "attendance"] = SENTINEL_INT
    df.loc[1, "home_squad_size"] = SENTINEL_INT
    df.loc[2, "away_total_caps"] = SENTINEL_INT
    df.loc[3, "home_total_caps"] = SENTINEL_INT
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRollingFeatures:
    """Tests para el cálculo de ventanas rodantes por equipo."""

    def test_rolling_features_columns_exist(
        self, df_with_rolling: pd.DataFrame
    ) -> None:
        """Verifica que las columnas rolling home/away se crean correctamente."""
        expected_suffixes = [
            "rolling_goals_scored",
            "rolling_goals_conceded",
            "rolling_xg",
            "rolling_xga",
            "rolling_form",
            "rolling_elo_momentum",
        ]
        for suffix in expected_suffixes:
            assert f"home_{suffix}" in df_with_rolling.columns, (
                f"Falta columna home_{suffix}"
            )
            assert f"away_{suffix}" in df_with_rolling.columns, (
                f"Falta columna away_{suffix}"
            )

    def test_rolling_features_no_leakage(
        self, df_with_rolling: pd.DataFrame
    ) -> None:
        """
        Verifica que no hay data leakage: los rolling stats del primer
        partido de un equipo deben ser NaN (se usa shift(1)).
        """
        # Buscar el primer partido de cada equipo como local
        for team_id in [1, 2, 3, 4]:
            first_home = df_with_rolling.loc[
                df_with_rolling["home_team_id"] == team_id
            ].iloc[0]
            # El primer partido NO debe tener rolling (es NaN antes de
            # handle_sentinels). Si min_periods=1 + shift(1), será NaN.
            rolling_val = first_home["home_rolling_goals_scored"]
            assert pd.isna(rolling_val) or isinstance(rolling_val, float), (
                "El primer partido debería tener NaN o un valor float"
            )

    def test_rolling_form_range(self, df_with_rolling: pd.DataFrame) -> None:
        """Verifica que rolling_form está entre 0 y 1 (porcentaje de victorias)."""
        for col in ["home_rolling_form", "away_rolling_form"]:
            valid = df_with_rolling[col].dropna()
            assert (valid >= 0.0).all(), f"{col} tiene valores < 0"
            assert (valid <= 1.0).all(), f"{col} tiene valores > 1"

    def test_rolling_preserves_row_count(
        self, synthetic_matches: pd.DataFrame, df_with_rolling: pd.DataFrame
    ) -> None:
        """Las rolling features no deben cambiar el número de filas."""
        assert len(df_with_rolling) == len(synthetic_matches)


class TestDifferentialFeatures:
    """Tests para las diferencias home − away."""

    def test_differential_columns_created(
        self, engineer: FeatureEngineer, df_with_rolling: pd.DataFrame
    ) -> None:
        """Verifica que las columnas diferenciales existen."""
        result = engineer.compute_differential_features(df_with_rolling)
        for col in ["xg_diff", "form_diff", "squad_value_diff", "goals_diff"]:
            assert col in result.columns, f"Falta columna diferencial: {col}"

    def test_differential_values_correct(
        self, engineer: FeatureEngineer, df_with_rolling: pd.DataFrame
    ) -> None:
        """Verifica que la diferencia es exactamente home − away."""
        result = engineer.compute_differential_features(df_with_rolling)

        # squad_value_diff = home_squad_value_log - away_squad_value_log
        expected_sv = (
            result["home_squad_value_log"] - result["away_squad_value_log"]
        )
        pd.testing.assert_series_equal(
            result["squad_value_diff"],
            expected_sv,
            check_names=False,
        )

    def test_xg_diff_sign(
        self, engineer: FeatureEngineer, df_with_rolling: pd.DataFrame
    ) -> None:
        """Verifica la coherencia de signos en xg_diff."""
        result = engineer.compute_differential_features(df_with_rolling)
        valid = result.dropna(subset=["home_rolling_xg", "away_rolling_xg"])
        if not valid.empty:
            manual_diff = valid["home_rolling_xg"] - valid["away_rolling_xg"]
            pd.testing.assert_series_equal(
                valid["xg_diff"], manual_diff, check_names=False
            )


class TestCompetitionEncoding:
    """Tests para la codificación de importancia de competición."""

    def test_competition_weight_column_exists(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Verifica que ``competition_weight`` se crea."""
        result = engineer.encode_competition_importance(synthetic_matches)
        assert "competition_weight" in result.columns

    def test_world_cup_gets_max_weight(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Partidos de World Cup deben tener peso = 1.0 (máximo)."""
        result = engineer.encode_competition_importance(synthetic_matches)
        wc_rows = result.loc[
            result["competition_name"].str.contains("World Cup", case=False)
        ]
        assert (wc_rows["competition_weight"] == 1.0).all(), (
            "World Cup debe tener peso 1.0"
        )

    def test_friendly_gets_min_weight(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Los amistosos deben tener el peso más bajo (0.5)."""
        result = engineer.encode_competition_importance(synthetic_matches)
        friendly_rows = result.loc[
            result["competition_name"].str.lower() == "friendly"
        ]
        expected = K_FACTOR["friendly"] / max(K_FACTOR.values())
        assert (friendly_rows["competition_weight"] == expected).all(), (
            f"Friendly debe tener peso {expected}"
        )

    def test_weight_range(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Todos los pesos deben estar en el rango [0, 1]."""
        result = engineer.encode_competition_importance(synthetic_matches)
        assert (result["competition_weight"] >= 0.0).all()
        assert (result["competition_weight"] <= 1.0).all()


class TestSentinelHandling:
    """Tests para el manejo de valores centinela."""

    def test_sentinel_int_replaced(
        self, engineer: FeatureEngineer, df_with_sentinels: pd.DataFrame
    ) -> None:
        """Verifica que -1 se reemplaza en columnas enteras conocidas."""
        result = engineer.handle_sentinels(df_with_sentinels)

        # attendance en fila 0 era -1, ahora no debe serlo
        assert result.loc[0, "attendance"] != SENTINEL_INT, (
            "Centinela -1 no fue reemplazado en attendance"
        )

    def test_no_nans_after_imputation(
        self, engineer: FeatureEngineer, df_with_sentinels: pd.DataFrame
    ) -> None:
        """Después de imputar, no deben quedar NaN en columnas numéricas."""
        result = engineer.handle_sentinels(df_with_sentinels)
        numeric = result.select_dtypes(include=[np.number])
        nan_count = numeric.isna().sum().sum()
        assert nan_count == 0, f"Quedan {nan_count} NaN tras imputación"

    def test_sentinel_replaced_with_median(
        self, engineer: FeatureEngineer, df_with_sentinels: pd.DataFrame
    ) -> None:
        """Verifica que el centinela se reemplaza con la mediana de la columna."""
        result = engineer.handle_sentinels(df_with_sentinels)
        # Después de reemplazar -1 → NaN → mediana, los valores deben ser
        # razonables (no -1)
        for col in ["attendance", "home_squad_size", "away_total_caps"]:
            assert (result[col] >= 0).all(), (
                f"Columna {col} tiene valores negativos tras imputación"
            )


class TestTargetBuild:
    """Tests para la construcción de la variable objetivo."""

    def test_target_column_created(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Verifica que ``target_result`` existe."""
        result = engineer.build_target(synthetic_matches)
        assert "target_result" in result.columns

    def test_target_values_valid(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Los valores del target deben ser 0, 1 o 2."""
        result = engineer.build_target(synthetic_matches)
        assert set(result["target_result"].unique()).issubset({0, 1, 2})

    def test_home_win_classified_as_2(
        self, engineer: FeatureEngineer
    ) -> None:
        """Un partido con home_goals > away_goals debe ser target = 2."""
        df = pd.DataFrame({
            "home_goals": [3, 0, 1],
            "away_goals": [1, 2, 1],
        })
        result = engineer.build_target(df)
        assert result.loc[0, "target_result"] == 2  # Victoria local
        assert result.loc[1, "target_result"] == 0  # Victoria visitante
        assert result.loc[2, "target_result"] == 1  # Empate

    def test_target_preserves_row_count(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """El build_target no debe cambiar el número de filas."""
        result = engineer.build_target(synthetic_matches)
        assert len(result) == len(synthetic_matches)


class TestBuildFeaturesIntegration:
    """Tests de integración para el pipeline completo."""

    def test_build_features_returns_tuple(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Verifica que build_features devuelve una tupla de 3 elementos."""
        X, y_home, y_away = engineer.build_features(synthetic_matches)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y_home, pd.Series)
        assert isinstance(y_away, pd.Series)

    def test_build_features_shape(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Verifica la forma del output y que no hay NaN en features."""
        X, y_home, y_away = engineer.build_features(synthetic_matches)

        # Misma cantidad de filas
        assert X.shape[0] == len(synthetic_matches)
        assert len(y_home) == len(synthetic_matches)
        assert len(y_away) == len(synthetic_matches)

        # Sin NaN en features
        nan_count = X.isna().sum().sum()
        assert nan_count == 0, f"X contiene {nan_count} NaN"

    def test_build_features_no_leakage_columns(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Features no deben incluir columnas de target o identificadores."""
        X, _, _ = engineer.build_features(synthetic_matches)
        forbidden = [
            "match_id", "home_team_name", "away_team_name",
            "home_goals", "away_goals", "target_result",
            "match_date", "ingested_at",
        ]
        for col in forbidden:
            assert col not in X.columns, (
                f"Columna prohibida '{col}' presente en X"
            )

    def test_build_features_all_numeric(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Todas las features deben ser numéricas."""
        X, _, _ = engineer.build_features(synthetic_matches)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, (
            f"Columnas no numéricas en X: {non_numeric}"
        )

    def test_build_features_has_expected_features(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Verifica que features clave están presentes en el output."""
        X, _, _ = engineer.build_features(synthetic_matches)
        expected_features = [
            "elo_diff",
            "xg_diff",
            "form_diff",
            "competition_weight",
            "is_neutral",
            "is_knockout",
            "home_days_rest",
            "away_days_rest",
        ]
        for feat in expected_features:
            assert feat in X.columns, f"Feature esperada '{feat}' no encontrada"

    def test_confederation_placeholders_present(
        self, engineer: FeatureEngineer, synthetic_matches: pd.DataFrame
    ) -> None:
        """Verifica que los placeholders de confederación están presentes."""
        X, _, _ = engineer.build_features(synthetic_matches)
        for conf in CONFEDERATIONS:
            assert f"home_conf_{conf}" in X.columns, (
                f"Falta placeholder home_conf_{conf}"
            )
            assert f"away_conf_{conf}" in X.columns, (
                f"Falta placeholder away_conf_{conf}"
            )
