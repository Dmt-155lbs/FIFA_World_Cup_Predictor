"""
Capa de calibración de probabilidades 1X2.

¿Por qué?
    El modelo XGBoost ``count:poisson`` → matriz de Poisson bivariado produce
    probabilidades 1X2 (local/empate/visitante) **sub-dispersas**: por la fuerte
    dependencia en ``fifa_attack_diff`` y la tendencia natural de los árboles a
    regresar a la media, sobre-estima al underdog y al empate. Eso genera EVs
    falsamente altos frente a las casas de apuestas (un mercado líquido y
    eficiente). La calibración aprende, con los resultados HISTÓRICOS reales, a
    "suavizar" esas probabilidades crudas hacia su frecuencia empírica.

Diseño:
    - Es una capa que mapea ``P_cruda[home,draw,away] → P_calibrada``. NO toca
      las λ (goles esperados): éstas siguen crudas para el ``BracketEngine`` y la
      simulación Monte Carlo. La calibración se aplica SÓLO en la capa de
      evaluación financiera y en los Value Bets del dashboard.
    - Se ajusta con las predicciones **out-of-fold** del walk-forward (cada
      predicción se hizo sobre datos que el modelo no vio al entrenar ese fold),
      evitando el sesgo de calibrar sobre datos de entrenamiento.

Métodos soportados:
    - ``isotonic`` (por defecto): regresión isotónica one-vs-rest por clase
      (igual que ``CalibratedClassifierCV(method='isotonic')`` internamente) y
      renormalización. No paramétrica y monótona; ideal para corregir una
      miscalibración no lineal como la sobre-estimación del underdog.
    - ``sigmoid`` (Platt): regresión logística multinomial sobre los log-probs
      crudos (vector/temperature scaling). Más suave, robusta con pocos datos.
"""
from __future__ import annotations

from typing import Any, Sequence

import joblib
import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

log = structlog.get_logger(__name__)

# Orden canónico de las clases 1X2 en TODO el sistema.
CLASSES: tuple[str, str, str] = ("home", "draw", "away")
_CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
_EPS = 1e-9


class ProbabilityCalibrator:
    """Calibra probabilidades 1X2 crudas contra resultados históricos reales.

    Parameters
    ----------
    method : str
        ``"isotonic"`` (default) o ``"sigmoid"`` (Platt multinomial).
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(
                f"método de calibración no soportado: {method!r} "
                "(use 'isotonic' o 'sigmoid')"
            )
        self.method = method
        self._iso: dict[str, IsotonicRegression] = {}
        self._lr: LogisticRegression | None = None
        self.fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Utilidades de formato                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_array(probs: Any) -> np.ndarray:
        """Acepta ``np.ndarray (N,3)`` o lista de dicts ``{home,draw,away}``."""
        if isinstance(probs, np.ndarray):
            arr = probs.astype(float)
        elif len(probs) and isinstance(probs[0], dict):
            arr = np.array(
                [[p["home"], p["draw"], p["away"]] for p in probs], dtype=float
            )
        else:
            arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """Re-normaliza cada fila a suma 1 (defensivo)."""
        s = X.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return X / s

    @staticmethod
    def _outcomes_to_idx(outcomes: Sequence[Any]) -> np.ndarray:
        return np.array(
            [_CLASS_IDX[str(o)] if not isinstance(o, (int, np.integer)) else int(o)
             for o in outcomes],
            dtype=int,
        )

    # ------------------------------------------------------------------ #
    #  Ajuste                                                             #
    # ------------------------------------------------------------------ #
    def fit(self, raw_probs: Any, outcomes: Sequence[Any]) -> "ProbabilityCalibrator":
        """Ajusta el calibrador.

        Parameters
        ----------
        raw_probs : array (N,3) o lista de dicts
            Probabilidades 1X2 crudas (out-of-fold).
        outcomes : secuencia
            Resultado real por partido: ``'home'/'draw'/'away'`` o índice 0/1/2.
        """
        X = self._normalize(self._to_array(raw_probs))
        y = self._outcomes_to_idx(outcomes)

        if self.method == "isotonic":
            for j, cls in enumerate(CLASSES):
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
                )
                iso.fit(X[:, j], (y == j).astype(float))
                self._iso[cls] = iso
        else:  # sigmoid / Platt multinomial sobre log-probs
            feats = np.log(np.clip(X, 1e-6, 1.0))
            self._lr = LogisticRegression(max_iter=1000, C=1e6)
            self._lr.fit(feats, y)

        self.fitted = True
        log.info(
            "Calibrador ajustado",
            metodo=self.method,
            n_muestras=int(X.shape[0]),
        )
        return self

    # ------------------------------------------------------------------ #
    #  Predicción                                                         #
    # ------------------------------------------------------------------ #
    def predict(self, raw_probs: Any) -> np.ndarray:
        """Devuelve las probabilidades calibradas (N,3) renormalizadas.

        Si no está ajustado, devuelve las crudas normalizadas (no-op seguro).
        """
        X = self._normalize(self._to_array(raw_probs))
        if not self.fitted:
            return X

        if self.method == "isotonic":
            cols = [self._iso[c].predict(X[:, j]) for j, c in enumerate(CLASSES)]
            cal = np.clip(np.vstack(cols).T, _EPS, None)
        else:
            feats = np.log(np.clip(X, 1e-6, 1.0))
            proba = self._lr.predict_proba(feats)
            cal = np.full((proba.shape[0], 3), _EPS)
            for k, c in enumerate(self._lr.classes_):
                cal[:, int(c)] = proba[:, k]

        return cal / cal.sum(axis=1, keepdims=True)

    def predict_one(
        self, p_home: float, p_draw: float, p_away: float
    ) -> tuple[float, float, float]:
        """Calibra una sola terna de probabilidades 1X2."""
        cal = self.predict(np.array([[p_home, p_draw, p_away]], dtype=float))[0]
        return float(cal[0]), float(cal[1]), float(cal[2])

    # ------------------------------------------------------------------ #
    #  Persistencia                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        joblib.dump(
            {
                "method": self.method,
                "iso": self._iso,
                "lr": self._lr,
                "fitted": self.fitted,
            },
            path,
        )
        log.info("Calibrador persistido", ruta=path, metodo=self.method)

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        data = joblib.load(path)
        obj = cls(method=data["method"])
        obj._iso = data.get("iso", {})
        obj._lr = data.get("lr")
        obj.fitted = data.get("fitted", False)
        return obj


def brier_multiclass(probs: np.ndarray, outcomes: Sequence[Any]) -> float:
    """Brier score multiclase (promedio sobre las 3 clases). Menor = mejor."""
    P = ProbabilityCalibrator._normalize(ProbabilityCalibrator._to_array(probs))
    y = ProbabilityCalibrator._outcomes_to_idx(outcomes)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)) / 3.0)
