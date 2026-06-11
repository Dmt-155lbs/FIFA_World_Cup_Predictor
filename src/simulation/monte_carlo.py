"""
Simulador Monte Carlo para el torneo FIFA 2026.

Ejecuta N simulaciones completas del torneo usando el ``BracketEngine``
y agrega los resultados para calcular probabilidades de campeonato,
avance a cada ronda, y estadísticas de goles por equipo.

Cada simulación usa un generador de números aleatorios derivado de una
``SeedSequence`` para garantizar reproducibilidad total independiente
del orden de ejecución.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from src.simulation.bracket_engine import BracketEngine

# ── Logger estructurado del módulo ──────────────────────────────────────────
logger = structlog.get_logger(__name__)


# ============================================================================ #
#  DATACLASS DE RESULTADOS AGREGADOS                                            #
# ============================================================================ #


@dataclass
class SimulationResults:
    """Resultados agregados de las N simulaciones Monte Carlo.

    Atributos
    ---------
    n_simulations : int
        Número total de simulaciones ejecutadas.
    champion_probs : dict[str, float]
        Probabilidad de ser campeón para cada equipo.
    finalist_probs : dict[str, float]
        Probabilidad de llegar a la final.
    semifinalist_probs : dict[str, float]
        Probabilidad de llegar a semifinales.
    round_advance_probs : dict[str, dict[str, float]]
        Probabilidad de avanzar a cada ronda, por equipo.
        Formato: ``{ronda: {equipo: probabilidad}}``.
    group_stage_probs : dict[str, dict[str, float]]
        Probabilidades de la fase de grupos por equipo.
        Formato: ``{equipo: {1ro: prob, 2do: prob, 3ro_clasif: prob, ...}}``.
    avg_goals_scored : dict[str, float]
        Promedio de goles marcados por equipo en el torneo.
    """

    n_simulations: int
    champion_probs: dict[str, float] = field(default_factory=dict)
    finalist_probs: dict[str, float] = field(default_factory=dict)
    semifinalist_probs: dict[str, float] = field(default_factory=dict)
    round_advance_probs: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    group_stage_probs: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    avg_goals_scored: dict[str, float] = field(default_factory=dict)


# ============================================================================ #
#  SIMULADOR MONTE CARLO                                                        #
# ============================================================================ #


class MonteCarloSimulator:
    """Simulador Monte Carlo para el bracket completo del Mundial 2026.

    Ejecuta múltiples simulaciones del torneo usando el ``BracketEngine``
    y agrega las frecuencias de cada resultado para estimar probabilidades.

    Parámetros
    ----------
    bracket_engine : BracketEngine
        Motor del bracket ya inicializado con la config YAML.
    seed : int
        Semilla raíz para reproducibilidad.  Se usa ``SeedSequence``
        para derivar generadores independientes por simulación.
    """

    def __init__(
        self,
        bracket_engine: BracketEngine,
        seed: int = 42,
    ) -> None:
        """Inicializa el simulador con un motor de bracket y semilla.

        Parámetros
        ----------
        bracket_engine : BracketEngine
            Instancia del motor de bracket configurado.
        seed : int
            Semilla base para el generador de números aleatorios.
        """
        self._engine: BracketEngine = bracket_engine
        self._seed: int = seed
        self._root_rng: np.random.Generator = np.random.default_rng(seed)

        logger.info(
            "Simulador Monte Carlo inicializado",
            seed=seed,
        )

    # ================================================================== #
    #  MÉTODO PRINCIPAL DE SIMULACIÓN                                      #
    # ================================================================== #

    def simulate(
        self,
        team_lambdas: dict[str, tuple[float, float]],
        n_simulations: int = 10_000,
    ) -> SimulationResults:
        """Ejecuta N simulaciones completas del torneo.

        Para cada simulación, crea un generador de números aleatorios
        independiente derivado de ``SeedSequence`` y ejecuta
        ``bracket_engine.simulate_full_bracket()``.  Al finalizar,
        agrega todos los resultados.

        Parámetros
        ----------
        team_lambdas : dict[str, tuple[float, float]]
            Lambdas ``{equipo: (ataque, defensa)}`` para todos los equipos.
        n_simulations : int
            Número de simulaciones Monte Carlo a ejecutar.

        Retorna
        -------
        SimulationResults
            Probabilidades y estadísticas agregadas.
        """
        logger.info(
            "Iniciando simulación Monte Carlo",
            n_simulaciones=n_simulations,
            equipos=len(team_lambdas),
        )

        # Crear SeedSequence para generar semillas independientes
        seed_seq = np.random.SeedSequence(self._seed)
        child_seeds: list[np.random.SeedSequence] = seed_seq.spawn(
            n_simulations
        )

        all_results: list[dict[str, Any]] = []
        intervalo_log: int = max(1, n_simulations // 10)

        for i in range(n_simulations):
            # Generador independiente para esta simulación
            sim_rng: np.random.Generator = np.random.default_rng(
                child_seeds[i]
            )

            # Simular torneo completo
            resultado = self._engine.simulate_full_bracket(
                team_lambdas=team_lambdas,
                rng=sim_rng,
            )
            all_results.append(resultado)

            # Log de progreso cada 10%
            if (i + 1) % intervalo_log == 0:
                porcentaje: float = ((i + 1) / n_simulations) * 100
                logger.info(
                    "Progreso de simulación",
                    completado=f"{porcentaje:.0f}%",
                    simulacion=i + 1,
                    de=n_simulations,
                )

        # Agregar resultados
        resultados_finales = self._aggregate_results(
            all_results, n_simulations, team_lambdas
        )

        logger.info(
            "Simulación Monte Carlo completada",
            n_simulaciones=n_simulations,
            equipos_con_prob_campeon=len(resultados_finales.champion_probs),
        )

        return resultados_finales

    # ================================================================== #
    #  AGREGACIÓN DE RESULTADOS                                            #
    # ================================================================== #

    def _aggregate_results(
        self,
        all_results: list[dict[str, Any]],
        n_simulations: int,
        team_lambdas: dict[str, tuple[float, float]],
    ) -> SimulationResults:
        """Agrega los resultados de todas las simulaciones.

        Cuenta frecuencias de campeonatos, finales, semifinales y avances
        por ronda, y las convierte en probabilidades dividiendo por
        ``n_simulations``.

        Parámetros
        ----------
        all_results : list[dict]
            Lista de resultados de cada simulación individual.
        n_simulations : int
            Número total de simulaciones (denominador).
        team_lambdas : dict[str, tuple[float, float]]
            Lambdas originales (para obtener la lista de equipos).

        Retorna
        -------
        SimulationResults
            Resultados agregados con probabilidades normalizadas.
        """
        # Contadores de frecuencia
        champion_count: dict[str, int] = defaultdict(int)
        finalist_count: dict[str, int] = defaultdict(int)
        semifinalist_count: dict[str, int] = defaultdict(int)
        round_advance_count: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        group_position_count: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        total_goals: dict[str, int] = defaultdict(int)
        total_matches: dict[str, int] = defaultdict(int)

        for resultado in all_results:
            champion: str = resultado["champion"]
            finalist: str = resultado["finalist"]
            semifinalistas: list[str] = resultado.get("semifinalists", [])

            # Campeonato y final
            if champion:
                champion_count[champion] += 1
            if finalist:
                finalist_count[finalist] += 1

            # Semifinalistas
            for sf in semifinalistas:
                semifinalist_count[sf] += 1

            # Avances por ronda
            round_advances: dict[str, list[str]] = resultado.get(
                "round_advances", {}
            )
            for ronda, equipos in round_advances.items():
                for equipo in equipos:
                    round_advance_count[ronda][equipo] += 1

            # Posiciones en fase de grupos
            group_standings = resultado.get("group_standings", {})
            for grupo_id, tabla in group_standings.items():
                for posicion_idx, equipo_data in enumerate(tabla):
                    nombre: str = equipo_data["equipo"]
                    posicion_label: str = f"posicion_{posicion_idx + 1}"
                    group_position_count[nombre][posicion_label] += 1

                    # Acumular goles del grupo
                    total_goals[nombre] += equipo_data.get("gf", 0)
                    total_matches[nombre] += equipo_data.get("partidos", 0)

            # Goles en fase eliminatoria
            all_match_results = resultado.get("all_results", {})
            for match_id, match_data in all_match_results.items():
                local_eq: str = match_data.get("local", "")
                visit_eq: str = match_data.get("visitante", "")
                gl: int = match_data.get("goles_local", 0)
                gv: int = match_data.get("goles_visitante", 0)
                if local_eq:
                    total_goals[local_eq] += gl
                    total_matches[local_eq] += 1
                if visit_eq:
                    total_goals[visit_eq] += gv
                    total_matches[visit_eq] += 1

        # ── Convertir frecuencias a probabilidades ──────────────────
        champion_probs: dict[str, float] = {
            eq: count / n_simulations
            for eq, count in champion_count.items()
        }
        finalist_probs: dict[str, float] = {
            eq: count / n_simulations
            for eq, count in finalist_count.items()
        }
        semifinalist_probs: dict[str, float] = {
            eq: count / n_simulations
            for eq, count in semifinalist_count.items()
        }

        # Probabilidades de avance por ronda
        round_advance_probs: dict[str, dict[str, float]] = {}
        for ronda, equipos_dict in round_advance_count.items():
            round_advance_probs[ronda] = {
                eq: count / n_simulations
                for eq, count in equipos_dict.items()
            }

        # Probabilidades de posición en grupo
        group_stage_probs: dict[str, dict[str, float]] = {}
        for equipo, posiciones in group_position_count.items():
            group_stage_probs[equipo] = {
                pos: count / n_simulations
                for pos, count in posiciones.items()
            }

        # Promedio de goles por equipo
        avg_goals: dict[str, float] = {}
        for equipo in team_lambdas:
            total_g = total_goals.get(equipo, 0)
            total_m = total_matches.get(equipo, 0)
            # Promedio por simulación, no por partido
            avg_goals[equipo] = total_g / n_simulations if n_simulations > 0 else 0.0

        return SimulationResults(
            n_simulations=n_simulations,
            champion_probs=champion_probs,
            finalist_probs=finalist_probs,
            semifinalist_probs=semifinalist_probs,
            round_advance_probs=round_advance_probs,
            group_stage_probs=group_stage_probs,
            avg_goals_scored=avg_goals,
        )
