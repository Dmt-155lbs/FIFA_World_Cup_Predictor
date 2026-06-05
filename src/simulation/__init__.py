"""
Módulo de simulación del torneo FIFA 2026.

Provee el motor de bracket y el simulador Monte Carlo para
generar probabilidades de avance y campeonato de cada equipo.
"""

from simulation.bracket_engine import BracketEngine
from simulation.monte_carlo import MonteCarloSimulator

__all__: list[str] = ["BracketEngine", "MonteCarloSimulator"]
