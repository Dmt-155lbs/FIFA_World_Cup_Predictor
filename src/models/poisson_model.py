"""
Modelo Poisson bivariado con corrección Dixon-Coles para partidos de fútbol.

Implementa la distribución de Poisson bivariada con la corrección de
dependencia de Dixon & Coles (1997) para modelar la correlación entre
goles locales y visitantes en marcadores bajos (0-0, 1-0, 0-1, 1-1).

La corrección ajusta las probabilidades independientes mediante un factor
tau(x, y, λ₁, λ₂, ρ) que captura la tendencia de los partidos reales
a producir resultados con pocos goles con mayor/menor frecuencia de
lo que predice un modelo Poisson independiente.

Referencia: Dixon, M. J. & Coles, S. G. (1997).
            "Modelling Association Football Scores and Inefficiencies
            in the Football Betting Market."

Autor: Mundial 2026 Team
"""

from __future__ import annotations

import numpy as np
import structlog
from scipy import optimize, stats

from src.utils.constants import MAX_GOALS_MATRIX

log = structlog.get_logger(__name__)


class BivariatePoisson:
    """
    Distribución de Poisson bivariada con corrección Dixon-Coles.

    Genera matrices de probabilidad de marcadores (goles local × goles
    visitante) a partir de las tasas esperadas (λ) de cada equipo,
    aplicando la corrección de dependencia ρ para marcadores bajos.

    Atributos:
        max_goals: Número máximo de goles a considerar en cada dimensión
                  de la matriz (tamaño = max_goals + 1).
        rho: Parámetro de dependencia Dixon-Coles. Valores negativos
            indican que los empates a bajo marcador son más probables
            de lo que predice el modelo independiente.
    """

    def __init__(
        self,
        max_goals: int = MAX_GOALS_MATRIX,
        rho: float = -0.13,
    ) -> None:
        """
        Inicializa el modelo con parámetros de la distribución.

        Args:
            max_goals: Goles máximos por equipo en la matriz de marcadores.
                      La matriz resultante será de tamaño (max_goals+1) ×
                      (max_goals+1). Por defecto usa MAX_GOALS_MATRIX=8.
            rho: Parámetro de dependencia Dixon-Coles (ρ). Valor típico
                entre -0.3 y 0.0. El valor -0.13 es el estimado empírico
                histórico para partidos internacionales.
        """
        self.max_goals: int = max_goals
        self.rho: float = rho

        log.info(
            "Modelo BivariatePoisson inicializado",
            max_goals=self.max_goals,
            rho=self.rho,
        )

    @staticmethod
    def _tau(
        x: int,
        y: int,
        lambda_home: float,
        lambda_away: float,
        rho: float,
    ) -> float:
        """
        Calcula el factor de corrección Dixon-Coles τ(x, y, λ₁, λ₂, ρ).

        Solo modifica las probabilidades para marcadores bajos:
        (0,0), (1,0), (0,1), (1,1). Para el resto retorna 1.0.

        Args:
            x: Goles del equipo local.
            y: Goles del equipo visitante.
            lambda_home: Tasa esperada de goles del equipo local (λ₁).
            lambda_away: Tasa esperada de goles del equipo visitante (λ₂).
            rho: Parámetro de dependencia (ρ).

        Returns:
            Factor multiplicativo τ para la celda (x, y).
        """
        if x == 0 and y == 0:
            return 1.0 - lambda_home * lambda_away * rho
        if x == 1 and y == 0:
            return 1.0 + lambda_away * rho
        if x == 0 and y == 1:
            return 1.0 + lambda_home * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def score_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float | None = None,
    ) -> np.ndarray:
        """
        Genera la matriz de probabilidades de marcadores con corrección Dixon-Coles.

        Calcula la distribución conjunta P(X=i, Y=j) donde X e Y son los
        goles de local y visitante respectivamente. Primero computa la
        distribución independiente y luego aplica la corrección τ.

        Args:
            lambda_home: Tasa esperada de goles del equipo local (λ₁ > 0).
            lambda_away: Tasa esperada de goles del equipo visitante (λ₂ > 0).
            rho: Parámetro de dependencia. Si es None, usa self.rho.

        Returns:
            Matriz numpy de forma (max_goals+1, max_goals+1) donde la
            celda [i, j] contiene P(home=i, away=j). La matriz está
            normalizada para que sume exactamente 1.0.
        """
        if rho is None:
            rho = self.rho

        n = self.max_goals + 1

        # Distribuciones marginales de Poisson independientes
        home_probs = stats.poisson.pmf(np.arange(n), mu=lambda_home)
        away_probs = stats.poisson.pmf(np.arange(n), mu=lambda_away)

        # Producto externo: distribución conjunta independiente
        matrix = np.outer(home_probs, away_probs)

        # Aplicar corrección Dixon-Coles a marcadores bajos
        for i in range(min(2, n)):
            for j in range(min(2, n)):
                tau = self._tau(i, j, lambda_home, lambda_away, rho)
                matrix[i, j] *= tau

        # Normalizar para que sume exactamente 1.0
        total = matrix.sum()
        if total > 0:
            matrix /= total

        log.debug(
            "Matriz de marcadores generada",
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
            suma_total=float(matrix.sum()),
        )

        return matrix

    def match_probabilities(
        self, matrix: np.ndarray
    ) -> dict[str, float]:
        """
        Calcula las probabilidades de victoria local, empate y victoria visitante.

        Args:
            matrix: Matriz de marcadores de forma (n, n) generada por
                   score_matrix().

        Returns:
            Diccionario con claves:
            - 'prob_home': Probabilidad de victoria del equipo local
                          (suma del triángulo inferior).
            - 'prob_draw': Probabilidad de empate (suma de la diagonal).
            - 'prob_away': Probabilidad de victoria del equipo visitante
                          (suma del triángulo superior).
        """
        n = matrix.shape[0]

        prob_home: float = 0.0
        prob_draw: float = 0.0
        prob_away: float = 0.0

        for i in range(n):
            for j in range(n):
                if i > j:
                    prob_home += matrix[i, j]
                elif i == j:
                    prob_draw += matrix[i, j]
                else:
                    prob_away += matrix[i, j]

        result = {
            "prob_home": float(prob_home),
            "prob_draw": float(prob_draw),
            "prob_away": float(prob_away),
        }

        log.debug(
            "Probabilidades de resultado calculadas",
            prob_home=result["prob_home"],
            prob_draw=result["prob_draw"],
            prob_away=result["prob_away"],
        )

        return result

    def over_under_probs(
        self,
        matrix: np.ndarray,
        threshold: float = 2.5,
    ) -> dict[str, float]:
        """
        Calcula probabilidades de over/under para un umbral de goles totales.

        Args:
            matrix: Matriz de marcadores de forma (n, n).
            threshold: Umbral de goles totales (por defecto 2.5, el más
                      común en apuestas deportivas).

        Returns:
            Diccionario con claves:
            - 'prob_over': P(goles_total > threshold)
            - 'prob_under': P(goles_total ≤ threshold)
        """
        n = matrix.shape[0]

        prob_over: float = 0.0
        prob_under: float = 0.0

        for i in range(n):
            for j in range(n):
                if (i + j) > threshold:
                    prob_over += matrix[i, j]
                else:
                    prob_under += matrix[i, j]

        result = {
            "prob_over": float(prob_over),
            "prob_under": float(prob_under),
        }

        log.debug(
            "Probabilidades over/under calculadas",
            threshold=threshold,
            prob_over=result["prob_over"],
            prob_under=result["prob_under"],
        )

        return result

    def expected_goals(
        self, matrix: np.ndarray
    ) -> tuple[float, float]:
        """
        Calcula los goles esperados marginales de cada equipo.

        Los goles esperados se computan como la esperanza de cada
        distribución marginal derivada de la matriz conjunta.

        Args:
            matrix: Matriz de marcadores de forma (n, n).

        Returns:
            Tupla (E[goles_local], E[goles_visitante]) con los goles
            esperados de cada equipo.
        """
        n = matrix.shape[0]
        goals_range = np.arange(n)

        # Distribuciones marginales
        marginal_home = matrix.sum(axis=1)  # Sumar sobre columnas (visitante)
        marginal_away = matrix.sum(axis=0)  # Sumar sobre filas (local)

        eg_home = float(np.dot(goals_range, marginal_home))
        eg_away = float(np.dot(goals_range, marginal_away))

        log.debug(
            "Goles esperados calculados",
            eg_home=eg_home,
            eg_away=eg_away,
        )

        return eg_home, eg_away

    def most_likely_scores(
        self,
        matrix: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[int, int, float]]:
        """
        Retorna los marcadores más probables ordenados por probabilidad.

        Args:
            matrix: Matriz de marcadores de forma (n, n).
            top_n: Número de marcadores a retornar (por defecto 5).

        Returns:
            Lista de tuplas (goles_local, goles_visitante, probabilidad)
            ordenada de mayor a menor probabilidad.
        """
        n = matrix.shape[0]

        # Recopilar todas las celdas con sus probabilidades
        scores: list[tuple[int, int, float]] = []
        for i in range(n):
            for j in range(n):
                scores.append((i, j, float(matrix[i, j])))

        # Ordenar por probabilidad descendente y tomar top_n
        scores.sort(key=lambda x: x[2], reverse=True)
        top_scores = scores[:top_n]

        log.debug(
            "Marcadores más probables calculados",
            top_n=top_n,
            marcador_1=f"{top_scores[0][0]}-{top_scores[0][1]}"
            if top_scores
            else "N/A",
        )

        return top_scores

    def fit_rho(
        self,
        historical_lambdas: np.ndarray,
        historical_results: np.ndarray,
    ) -> float:
        """
        Estima el parámetro óptimo ρ a partir de datos históricos.

        Maximiza la log-verosimilitud del modelo Dixon-Coles sobre un
        conjunto de partidos históricos con lambdas predichos y resultados
        reales. Utiliza scipy.optimize.minimize con el método L-BFGS-B.

        Args:
            historical_lambdas: Array de forma (N, 2) donde cada fila es
                               [lambda_home, lambda_away] predichos.
            historical_results: Array de forma (N, 2) donde cada fila es
                               [goles_home_reales, goles_away_reales].

        Returns:
            Valor óptimo de ρ estimado por máxima verosimilitud.

        Raises:
            ValueError: Si los arrays tienen formas incompatibles.
        """
        if historical_lambdas.shape != historical_results.shape:
            raise ValueError(
                f"Formas incompatibles: lambdas={historical_lambdas.shape}, "
                f"resultados={historical_results.shape}"
            )
        if historical_lambdas.shape[1] != 2:
            raise ValueError(
                f"Se esperan 2 columnas, se recibieron "
                f"{historical_lambdas.shape[1]}"
            )

        def neg_log_likelihood(rho_arr: np.ndarray) -> float:
            """Calcula la log-verosimilitud negativa para un valor de ρ."""
            rho_val = float(rho_arr[0])
            total_ll: float = 0.0

            for idx in range(len(historical_lambdas)):
                lh = float(historical_lambdas[idx, 0])
                la = float(historical_lambdas[idx, 1])
                gh = int(historical_results[idx, 0])
                ga = int(historical_results[idx, 1])

                # Probabilidad Poisson independiente
                p_home = stats.poisson.pmf(gh, mu=lh)
                p_away = stats.poisson.pmf(ga, mu=la)

                # Factor de corrección Dixon-Coles
                tau = self._tau(gh, ga, lh, la, rho_val)

                prob = p_home * p_away * tau

                # Evitar log(0) con umbral mínimo
                prob = max(prob, 1e-15)
                total_ll += np.log(prob)

            return -total_ll

        # Optimización con límites en ρ ∈ [-0.5, 0.5]
        result = optimize.minimize(
            neg_log_likelihood,
            x0=np.array([self.rho]),
            method="L-BFGS-B",
            bounds=[(-0.5, 0.5)],
        )

        optimal_rho = float(result.x[0])
        self.rho = optimal_rho

        log.info(
            "Parámetro ρ estimado por máxima verosimilitud",
            rho_optimo=optimal_rho,
            log_verosimilitud=-result.fun,
            n_partidos=len(historical_lambdas),
            convergencia=result.success,
        )

        return optimal_rho
