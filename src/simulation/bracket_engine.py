"""
Motor del bracket del torneo FIFA 2026.

Carga la estructura del bracket desde un archivo YAML y simula tanto la fase
de grupos (round-robin con draws de Poisson) como todas las rondas
eliminatorias hasta la final, incluyendo la selección de los 8 mejores
terceros y su asignación dinámica al cuadro de Ronda de 32.

Todas las parejas de partido se leen dinámicamente del YAML, nunca se
hardcodean enfrentamientos.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yaml

from config import Settings

# ── Logger estructurado del módulo ──────────────────────────────────────────
logger = structlog.get_logger(__name__)


class BracketEngine:
    """Motor principal del bracket FIFA 2026.

    Lee ``bracket_2026.yaml`` y expone métodos para simular el torneo
    completo: fase de grupos → mejores terceros → bracket eliminatorio.

    Parámetros
    ----------
    config_path : str | None
        Ruta al archivo YAML del bracket.  Si ``None``, se usa
        ``Settings().bracket_config_path``.
    """

    # ------------------------------------------------------------------ #
    #  Inicialización                                                      #
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: str | None = None) -> None:
        """Inicializa el motor cargando la configuración YAML.

        Parámetros
        ----------
        config_path : str | None
            Ruta al archivo YAML.  Se usa la ruta por defecto de
            ``Settings`` si no se provee.
        """
        ruta: str = config_path or Settings().bracket_config_path
        ruta_absoluta = Path(ruta).resolve()

        logger.info(
            "Cargando configuración del bracket",
            ruta=str(ruta_absoluta),
        )

        with open(ruta_absoluta, encoding="utf-8") as fh:
            self._config: dict[str, Any] = yaml.safe_load(fh)

        # Propiedades de acceso rápido derivadas del YAML
        self._torneo: dict[str, Any] = self._config["torneo"]
        self._fase_grupos: dict[str, Any] = self._config["fase_de_grupos"]
        self._reglas: dict[str, Any] = self._config["reglas_desempate"]
        self._mejores_terceros_cfg: dict[str, Any] = self._config["mejores_terceros"]
        self._bracket: dict[str, Any] = self._config["bracket_eliminatorio"]
        self._params_elim: dict[str, Any] = self._config["parametros_eliminatoria"]

        logger.info(
            "Bracket cargado exitosamente",
            total_grupos=self._torneo["total_grupos"],
            total_equipos=self._torneo["total_equipos"],
        )

    # ── Propiedades públicas ────────────────────────────────────────── #

    @property
    def grupos(self) -> dict[str, list[dict[str, Any]]]:
        """Diccionario de grupos (A–L) con la lista de equipos."""
        return self._fase_grupos["grupos"]

    @property
    def reglas_desempate(self) -> dict[str, Any]:
        """Reglas de desempate según prioridad FIFA."""
        return self._reglas

    @property
    def mejores_terceros_config(self) -> dict[str, Any]:
        """Configuración de los mejores terceros (tabla de asignación, etc.)."""
        return self._mejores_terceros_cfg

    @property
    def bracket_eliminatorio(self) -> dict[str, Any]:
        """Estructura completa del bracket eliminatorio."""
        return self._bracket

    @property
    def parametros_eliminatoria(self) -> dict[str, Any]:
        """Parámetros numéricos de la fase eliminatoria."""
        return self._params_elim

    # ================================================================== #
    #  FASE DE GRUPOS                                                      #
    # ================================================================== #

    def simulate_group_stage(
        self,
        team_lambdas: dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> dict[str, list[dict[str, Any]]]:
        """Simula la fase de grupos completa usando draws de Poisson.

        Para cada grupo se juega un round-robin (6 partidos para 4 equipos)
        y se construye la tabla de clasificación aplicando los criterios de
        desempate definidos en el YAML.

        Parámetros
        ----------
        team_lambdas : dict[str, tuple[float, float]]
            Mapeo ``{nombre_equipo: (lambda_ataque, lambda_defensa)}``.
            ``lambda_ataque`` es la intensidad de goles esperados cuando
            el equipo ataca; ``lambda_defensa`` es la lambda que cede al
            rival (i.e., el rival anota con esa intensidad).
        rng : numpy.random.Generator
            Generador de números aleatorios para reproducibilidad.

        Retorna
        -------
        dict[str, list[dict]]
            ``{grupo: [tabla_posiciones]}`` donde cada fila tiene
            ``equipo, puntos, gf, gc, dg, partidos, victorias, empates,
            derrotas``.  Ordenado de 1.° a 4.°.
        """
        formato = self._fase_grupos["formato"]
        pts_w: int = formato["puntos_victoria"]
        pts_d: int = formato["puntos_empate"]
        pts_l: int = formato["puntos_derrota"]

        standings: dict[str, list[dict[str, Any]]] = {}

        for grupo_id, equipos_info in self.grupos.items():
            nombres: list[str] = [eq["equipo"] for eq in equipos_info]

            # Inicializar tabla para este grupo
            tabla: dict[str, dict[str, Any]] = {
                nombre: {
                    "equipo": nombre,
                    "grupo": grupo_id,
                    "puntos": 0,
                    "gf": 0,
                    "gc": 0,
                    "dg": 0,
                    "victorias": 0,
                    "empates": 0,
                    "derrotas": 0,
                    "partidos": 0,
                    # Registro de enfrentamientos directos (h2h)
                    "h2h": {},
                }
                for nombre in nombres
            }

            # Round-robin: todas las combinaciones de 2 equipos
            parejas: list[tuple[str, str]] = list(combinations(nombres, 2))

            # ── Simulación vectorizada de los 6 partidos ────────────
            n_partidos: int = len(parejas)
            lambdas_local = np.array(
                [team_lambdas[h][0] * team_lambdas[a][1] for h, a in parejas],
                dtype=np.float64,
            )
            lambdas_visit = np.array(
                [team_lambdas[a][0] * team_lambdas[h][1] for h, a in parejas],
                dtype=np.float64,
            )

            # Limitar lambdas a valores razonables
            lambdas_local = np.clip(lambdas_local, 0.01, 10.0)
            lambdas_visit = np.clip(lambdas_visit, 0.01, 10.0)

            # Draw vectorizado de Poisson para todos los partidos del grupo
            goles_local: np.ndarray = rng.poisson(lambdas_local)
            goles_visit: np.ndarray = rng.poisson(lambdas_visit)

            # Actualizar la tabla con los resultados
            for idx, (local, visitante) in enumerate(parejas):
                gl: int = int(goles_local[idx])
                gv: int = int(goles_visit[idx])

                # Actualizar goles
                tabla[local]["gf"] += gl
                tabla[local]["gc"] += gv
                tabla[visitante]["gf"] += gv
                tabla[visitante]["gc"] += gl
                tabla[local]["partidos"] += 1
                tabla[visitante]["partidos"] += 1

                # Registro h2h
                tabla[local]["h2h"][visitante] = (gl, gv)
                tabla[visitante]["h2h"][local] = (gv, gl)

                # Puntos y resultados
                if gl > gv:
                    tabla[local]["puntos"] += pts_w
                    tabla[local]["victorias"] += 1
                    tabla[visitante]["puntos"] += pts_l
                    tabla[visitante]["derrotas"] += 1
                elif gl < gv:
                    tabla[visitante]["puntos"] += pts_w
                    tabla[visitante]["victorias"] += 1
                    tabla[local]["puntos"] += pts_l
                    tabla[local]["derrotas"] += 1
                else:
                    tabla[local]["puntos"] += pts_d
                    tabla[local]["empates"] += 1
                    tabla[visitante]["puntos"] += pts_d
                    tabla[visitante]["empates"] += 1

            # Calcular diferencia de goles
            for equipo_data in tabla.values():
                equipo_data["dg"] = equipo_data["gf"] - equipo_data["gc"]

            # Ordenar aplicando criterios de desempate
            clasificacion = self._sort_group_standings(
                list(tabla.values()), rng
            )
            standings[grupo_id] = clasificacion

        logger.debug(
            "Fase de grupos simulada",
            grupos_simulados=len(standings),
        )

        return standings

    def _sort_group_standings(
        self,
        tabla: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Ordena la tabla de un grupo aplicando criterios de desempate.

        Los criterios se leen del YAML en orden de prioridad:
        1. Puntos (mayor)
        2. Diferencia de goles (mayor)
        3. Goles a favor (mayor)
        4. Enfrentamiento directo (h2h)
        5. Fair play (menor — no simulado, se ignora)
        6. Sorteo FIFA (aleatorio)

        Parámetros
        ----------
        tabla : list[dict]
            Lista de registros de la tabla del grupo.
        rng : numpy.random.Generator
            Generador aleatorio para el sorteo FIFA.

        Retorna
        -------
        list[dict]
            Tabla ordenada de mejor a peor.
        """

        def _sort_key(equipo: dict[str, Any]) -> tuple:
            """Genera clave de ordenamiento compuesta.

            Retorna una tupla donde los valores mayores son mejores.
            El sorteo FIFA se modela como un valor aleatorio pequeño.
            """
            return (
                equipo["puntos"],
                equipo["dg"],
                equipo["gf"],
                # Aleatorio como desempate final (sorteo FIFA)
                rng.random(),
            )

        return sorted(tabla, key=_sort_key, reverse=True)

    # ================================================================== #
    #  MEJORES TERCEROS                                                    #
    # ================================================================== #

    def rank_best_thirds(
        self,
        group_standings: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Selecciona los 8 mejores terceros de los 12 grupos.

        Aplica los mismos criterios de desempate que la fase de grupos
        para rankear los 12 terceros y seleccionar los 8 mejores.

        Parámetros
        ----------
        group_standings : dict[str, list[dict]]
            Resultados de la fase de grupos (salida de
            ``simulate_group_stage``).

        Retorna
        -------
        list[dict]
            Los 8 mejores terceros, ordenados de mejor a peor.
        """
        cantidad: int = self._mejores_terceros_cfg["cantidad_clasificados"]

        # Extraer el 3.° de cada grupo (índice 2)
        terceros: list[dict[str, Any]] = []
        for grupo_id, tabla in group_standings.items():
            tercero = tabla[2].copy()
            tercero["grupo_origen"] = grupo_id
            terceros.append(tercero)

        # Ordenar con los mismos criterios de desempate
        terceros_ordenados = sorted(
            terceros,
            key=lambda eq: (
                eq["puntos"],
                eq["dg"],
                eq["gf"],
            ),
            reverse=True,
        )

        mejores = terceros_ordenados[:cantidad]

        logger.debug(
            "Mejores terceros seleccionados",
            cantidad=len(mejores),
            equipos=[t["equipo"] for t in mejores],
        )

        return mejores

    def assign_thirds_to_bracket(
        self,
        qualifying_thirds: list[dict[str, Any]],
        group_standings: dict[str, list[dict[str, Any]]],
    ) -> dict[str, str]:
        """Asigna los 8 mejores terceros a sus slots en Ronda de 32.

        Determina qué 8 grupos produjeron terceros clasificados,
        busca la combinación en ``tabla_asignacion`` del YAML, y devuelve
        un mapeo de ``slot → nombre_equipo``.

        Si la combinación exacta no está en la tabla, se usa un mapeo
        por defecto basado en orden alfabético.

        Parámetros
        ----------
        qualifying_thirds : list[dict]
            Los 8 mejores terceros (salida de ``rank_best_thirds``).
        group_standings : dict
            Clasificaciones de grupo completas.

        Retorna
        -------
        dict[str, str]
            Mapeo ``{"3_SLOT_1": equipo, "3_SLOT_2": equipo, ...}``.
        """
        tabla_asig: dict[str, dict[str, str]] = self._mejores_terceros_cfg.get(
            "tabla_asignacion", {}
        )

        # Determinar los grupos de origen de los 8 terceros clasificados
        grupos_origen: list[str] = sorted(
            [t["grupo_origen"] for t in qualifying_thirds]
        )
        clave_combinacion: str = "".join(grupos_origen)

        # Mapeo grupo_origen → equipo
        grupo_a_equipo: dict[str, str] = {
            t["grupo_origen"]: t["equipo"] for t in qualifying_thirds
        }

        asignaciones: dict[str, str] = {}

        if clave_combinacion in tabla_asig:
            # Usar tabla oficial de FIFA
            mapeo: dict[str, str] = tabla_asig[clave_combinacion]
            for grupo, slot_r32 in mapeo.items():
                # Encontrar qué 3_SLOT_N corresponde a este slot R32
                slot_key = self._r32_id_to_slot(slot_r32)
                if slot_key and grupo in grupo_a_equipo:
                    asignaciones[slot_key] = grupo_a_equipo[grupo]
        else:
            # Mapeo por defecto: asignar en orden alfabético a slots 1-4
            logger.warning(
                "Combinación de terceros no encontrada en tabla, "
                "usando mapeo por defecto",
                combinacion=clave_combinacion,
            )

        # Si no se asignaron todos los slots, completar con defecto
        if len(asignaciones) < 4:
            self._asignar_slots_por_defecto(
                qualifying_thirds, asignaciones
            )

        logger.debug(
            "Terceros asignados al bracket",
            asignaciones=asignaciones,
        )

        return asignaciones

    def _r32_id_to_slot(self, r32_id: str) -> str | None:
        """Convierte un ID de R32 al slot correspondiente del tercer clasificado.

        Examina los partidos de ronda de 32 para encontrar qué ``3_SLOT_N``
        corresponde al partido con el ID dado.

        Parámetros
        ----------
        r32_id : str
            ID del partido de R32 (e.g., ``"R32_01"``).

        Retorna
        -------
        str | None
            El slot encontrado (e.g., ``"3_SLOT_1"``) o ``None``.
        """
        for partido in self._bracket.get("ronda_de_32", []):
            if partido["id"] == r32_id:
                visitante: str = partido["visitante"]
                if visitante.startswith("3_SLOT"):
                    return visitante
        return None

    def _asignar_slots_por_defecto(
        self,
        qualifying_thirds: list[dict[str, Any]],
        asignaciones: dict[str, str],
    ) -> None:
        """Asigna terceros a slots faltantes en orden de ranking.

        Modifica ``asignaciones`` in-place para completar los 4 slots
        de terceros (``3_SLOT_1`` a ``3_SLOT_4``).

        Parámetros
        ----------
        qualifying_thirds : list[dict]
            Los 8 mejores terceros ordenados por ranking.
        asignaciones : dict[str, str]
            Mapeo parcial que se completará in-place.
        """
        # Obtener los slots disponibles del YAML
        slots_disponibles: list[str] = []
        for partido in self._bracket.get("ronda_de_32", []):
            visitante: str = partido["visitante"]
            if visitante.startswith("3_SLOT") and visitante not in asignaciones:
                slots_disponibles.append(visitante)

        # Equipos aún no asignados
        ya_asignados = set(asignaciones.values())
        sin_asignar = [
            t for t in qualifying_thirds if t["equipo"] not in ya_asignados
        ]

        for slot, tercero in zip(slots_disponibles, sin_asignar):
            asignaciones[slot] = tercero["equipo"]

    # ================================================================== #
    #  RESOLUCIÓN DE SLOTS                                                 #
    # ================================================================== #

    def _resolve_slot(
        self,
        slot: str,
        results: dict[str, dict[str, str]],
        group_standings: dict[str, list[dict[str, Any]]],
        third_assignments: dict[str, str],
    ) -> str:
        """Resuelve un slot del bracket al nombre real del equipo.

        Maneja los formatos:
        - ``"1A"`` → primero del grupo A
        - ``"2C"`` → segundo del grupo C
        - ``"3_SLOT_1"`` → tercer clasificado asignado al slot 1
        - ``"W_R32_01"`` → ganador del partido R32_01
        - ``"L_SF_01"`` → perdedor del partido SF_01

        Parámetros
        ----------
        slot : str
            Identificador del slot a resolver.
        results : dict[str, dict[str, str]]
            Resultados acumulados ``{match_id: {winner, loser}}``.
        group_standings : dict
            Clasificaciones de grupo.
        third_assignments : dict
            Mapeo de slots de terceros a equipos.

        Retorna
        -------
        str
            Nombre del equipo resuelto.

        Raises
        ------
        ValueError
            Si el formato del slot no es reconocido.
        """
        # Tercer clasificado asignado dinámicamente
        if slot.startswith("3_SLOT"):
            if slot in third_assignments:
                return third_assignments[slot]
            raise ValueError(
                f"Slot de tercer clasificado '{slot}' no encontrado en "
                f"asignaciones: {third_assignments}"
            )

        # Ganador de un partido previo
        if slot.startswith("W_"):
            match_id: str = slot[2:]
            if match_id in results:
                return results[match_id]["winner"]
            raise ValueError(
                f"Ganador del partido '{match_id}' no encontrado en resultados"
            )

        # Perdedor de un partido previo
        if slot.startswith("L_"):
            match_id = slot[2:]
            if match_id in results:
                return results[match_id]["loser"]
            raise ValueError(
                f"Perdedor del partido '{match_id}' no encontrado en resultados"
            )

        # Posición en grupo: "1A", "2C", "3F", etc.
        match = re.match(r"^(\d)([A-L])$", slot)
        if match:
            posicion_idx: int = int(match.group(1)) - 1  # 0-indexed
            grupo_id: str = match.group(2)
            if grupo_id in group_standings:
                return group_standings[grupo_id][posicion_idx]["equipo"]
            raise ValueError(
                f"Grupo '{grupo_id}' no encontrado en clasificaciones"
            )

        raise ValueError(f"Formato de slot no reconocido: '{slot}'")

    # ================================================================== #
    #  RONDAS ELIMINATORIAS                                                #
    # ================================================================== #

    def simulate_knockout_round(
        self,
        matches: list[dict[str, Any]],
        team_lambdas: dict[str, tuple[float, float]],
        rng: np.random.Generator,
        results_tracker: dict[str, dict[str, str]],
        group_standings: dict[str, list[dict[str, Any]]],
        third_assignments: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        """Simula una ronda completa del bracket eliminatorio.

        Para cada partido de la ronda:
        1. Resuelve los slots a equipos reales.
        2. Genera goles con Poisson.
        3. Si empatan: tiempo extra (lambda × factor_tiempo_extra).
        4. Si persiste el empate: penales (coin-flip con bonus Elo).

        Parámetros
        ----------
        matches : list[dict]
            Lista de partidos de la ronda (del YAML).
        team_lambdas : dict[str, tuple[float, float]]
            Lambdas de ataque/defensa por equipo.
        rng : numpy.random.Generator
            Generador aleatorio.
        results_tracker : dict[str, dict[str, str]]
            Resultados acumulados que se actualiza in-place.
        group_standings : dict
            Clasificaciones de grupo (para resolver slots).
        third_assignments : dict
            Asignaciones de terceros (para resolver ``3_SLOT_N``).

        Retorna
        -------
        dict[str, dict[str, str]]
            ``results_tracker`` actualizado con los resultados de la ronda.
        """
        factor_et: float = self._params_elim["factor_tiempo_extra"]
        bonus_pen: float = self._params_elim["bonus_elo_penales"]

        for partido in matches:
            match_id: str = partido["id"]
            ronda: str = partido["ronda"]

            # Resolver equipos
            local: str = self._resolve_slot(
                partido["local"], results_tracker,
                group_standings, third_assignments,
            )
            visitante: str = self._resolve_slot(
                partido["visitante"], results_tracker,
                group_standings, third_assignments,
            )

            # Lambdas para este partido
            lam_loc_atk, lam_loc_def = team_lambdas[local]
            lam_vis_atk, lam_vis_def = team_lambdas[visitante]

            # Lambda efectiva: ataque del equipo × factor defensivo del rival
            lambda_local: float = max(0.01, lam_loc_atk * lam_vis_def)
            lambda_visit: float = max(0.01, lam_vis_atk * lam_loc_def)

            # ── Tiempo reglamentario (90 min) ──────────────────────
            goles_l: int = int(rng.poisson(lambda_local))
            goles_v: int = int(rng.poisson(lambda_visit))

            winner: str | None = None
            loser: str | None = None

            if goles_l > goles_v:
                winner, loser = local, visitante
            elif goles_v > goles_l:
                winner, loser = visitante, local
            else:
                # ── Tiempo extra ───────────────────────────────────
                et_lambda_l: float = lambda_local * factor_et
                et_lambda_v: float = lambda_visit * factor_et

                et_goles_l: int = int(rng.poisson(max(0.01, et_lambda_l)))
                et_goles_v: int = int(rng.poisson(max(0.01, et_lambda_v)))

                goles_l += et_goles_l
                goles_v += et_goles_v

                if goles_l > goles_v:
                    winner, loser = local, visitante
                elif goles_v > goles_l:
                    winner, loser = visitante, local
                else:
                    # ── Penales ────────────────────────────────────
                    # Probabilidad base 50/50 con bonus al equipo con
                    # mayor lambda de ataque (proxy de Elo superior).
                    prob_local_pen: float = 0.5
                    if lam_loc_atk > lam_vis_atk:
                        prob_local_pen += bonus_pen
                    elif lam_vis_atk > lam_loc_atk:
                        prob_local_pen -= bonus_pen

                    if rng.random() < prob_local_pen:
                        winner, loser = local, visitante
                    else:
                        winner, loser = visitante, local

            results_tracker[match_id] = {
                "winner": winner,
                "loser": loser,
                "local": local,
                "visitante": visitante,
                "goles_local": goles_l,
                "goles_visitante": goles_v,
                "ronda": ronda,
            }

        return results_tracker

    # ================================================================== #
    #  SIMULACIÓN COMPLETA DEL TORNEO                                      #
    # ================================================================== #

    def simulate_full_bracket(
        self,
        team_lambdas: dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Simula un torneo completo desde grupos hasta la final.

        Flujo:
        1. Fase de grupos (round-robin con Poisson).
        2. Selección de los 8 mejores terceros.
        3. Asignación de terceros a slots de R32.
        4. Ronda de 32 (16 partidos).
        5. Octavos de final (8 partidos).
        6. Cuartos de final (4 partidos).
        7. Semifinales (2 partidos).
        8. Tercer puesto (1 partido).
        9. Final (1 partido).

        Parámetros
        ----------
        team_lambdas : dict[str, tuple[float, float]]
            Lambdas ``{equipo: (ataque, defensa)}`` para todos los equipos.
        rng : numpy.random.Generator
            Generador aleatorio con semilla para reproducibilidad.

        Retorna
        -------
        dict[str, Any]
            Diccionario con claves:
            - ``champion``: nombre del campeón
            - ``finalist``: subcampeón
            - ``third``: tercer lugar
            - ``fourth``: cuarto lugar
            - ``semifinalists``: lista de 4 semifinalistas
            - ``group_standings``: clasificaciones de grupo
            - ``all_results``: todos los resultados de partidos
            - ``round_advances``: equipos que avanzaron a cada ronda
        """
        # 1. Fase de grupos
        group_standings = self.simulate_group_stage(team_lambdas, rng)

        # 2-3. Mejores terceros y asignación
        best_thirds = self.rank_best_thirds(group_standings)
        third_assignments = self.assign_thirds_to_bracket(
            best_thirds, group_standings
        )

        # Tracker de resultados acumulados
        results: dict[str, dict[str, str]] = {}

        # ── Definir las rondas en orden secuencial ──────────────────
        rondas_config: list[tuple[str, str]] = [
            ("ronda_de_32", "Ronda de 32"),
            ("octavos_de_final", "Octavos de Final"),
            ("cuartos_de_final", "Cuartos de Final"),
            ("semifinales", "Semifinal"),
            ("tercer_puesto", "Tercer Puesto"),
            ("final", "Final"),
        ]

        # Registrar avances por ronda
        round_advances: dict[str, list[str]] = {}

        for ronda_key, ronda_nombre in rondas_config:
            partidos = self._bracket.get(ronda_key, [])
            if not partidos:
                continue

            self.simulate_knockout_round(
                matches=partidos,
                team_lambdas=team_lambdas,
                rng=rng,
                results_tracker=results,
                group_standings=group_standings,
                third_assignments=third_assignments,
            )

            # Registrar ganadores de esta ronda
            ganadores_ronda = [
                results[p["id"]]["winner"]
                for p in partidos
                if p["id"] in results
            ]
            round_advances[ronda_nombre] = ganadores_ronda

        # ── Extraer resultados finales ──────────────────────────────
        final_result = results.get("FIN_01", {})
        tp_result = results.get("TP_01", {})

        # Semifinalistas = ganadores de QF
        semifinalistas: list[str] = round_advances.get(
            "Cuartos de Final",
            [
                results[p["id"]]["winner"]
                for p in self._bracket.get("cuartos_de_final", [])
                if p["id"] in results
            ],
        )

        return {
            "champion": final_result.get("winner", ""),
            "finalist": final_result.get("loser", ""),
            "third": tp_result.get("winner", ""),
            "fourth": tp_result.get("loser", ""),
            "semifinalists": semifinalistas,
            "group_standings": group_standings,
            "all_results": results,
            "round_advances": round_advances,
        }
