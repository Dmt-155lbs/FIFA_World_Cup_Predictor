"""
Datos semilla para las tablas dimensionales del sistema.

Contiene los 48 equipos clasificados al Mundial FIFA 2026 con sus códigos
FIFA y confederaciones, así como las competiciones históricas relevantes
para el feature engineering del modelo predictivo.

Estos datos se usan en la carga inicial (Day 0) para poblar DIM_TEAM y
DIM_COMPETITION antes de ejecutar la ingesta de datos históricos.
"""

from __future__ import annotations


# ============================================================================
# 48 EQUIPOS CLASIFICADOS — MUNDIAL FIFA 2026
# ============================================================================
# Formato: (team_name, fifa_code, confederation, fifa_ranking)
# Ranking FIFA actualizado a junio 2026.

WORLD_CUP_2026_TEAMS: list[tuple[str, str, str, int]] = [
    # ── CONMEBOL (6) ──────────────────────────────────────────────────
    ("Argentina", "ARG", "CONMEBOL", 1),
    ("Brazil", "BRA", "CONMEBOL", 5),
    ("Uruguay", "URU", "CONMEBOL", 11),
    ("Colombia", "COL", "CONMEBOL", 12),
    ("Ecuador", "ECU", "CONMEBOL", 30),
    ("Paraguay", "PAR", "CONMEBOL", 55),

    # ── UEFA (16) ─────────────────────────────────────────────────────
    ("France", "FRA", "UEFA", 2),
    ("Spain", "ESP", "UEFA", 3),
    ("England", "ENG", "UEFA", 4),
    ("Portugal", "POR", "UEFA", 6),
    ("Netherlands", "NED", "UEFA", 7),
    ("Belgium", "BEL", "UEFA", 8),
    ("Germany", "GER", "UEFA", 9),
    ("Italy", "ITA", "UEFA", 10),
    ("Croatia", "CRO", "UEFA", 13),
    ("Denmark", "DEN", "UEFA", 17),
    ("Switzerland", "SUI", "UEFA", 19),
    ("Austria", "AUT", "UEFA", 22),
    ("Serbia", "SRB", "UEFA", 33),
    ("Wales", "WAL", "UEFA", 27),
    ("Turkey", "TUR", "UEFA", 26),
    ("Ukraine", "UKR", "UEFA", 23),

    # ── CONCACAF (6 — incluye 3 anfitriones) ──────────────────────────
    ("United States", "USA", "CONCACAF", 14),
    ("Mexico", "MEX", "CONCACAF", 15),
    ("Canada", "CAN", "CONCACAF", 40),
    ("Jamaica", "JAM", "CONCACAF", 57),
    ("Panama", "PAN", "CONCACAF", 44),
    ("Honduras", "HON", "CONCACAF", 72),

    # ── AFC (8) ───────────────────────────────────────────────────────
    ("Japan", "JPN", "AFC", 16),
    ("South Korea", "KOR", "AFC", 21),
    ("Australia", "AUS", "AFC", 24),
    ("Iran", "IRN", "AFC", 20),
    ("Saudi Arabia", "KSA", "AFC", 56),
    ("Qatar", "QAT", "AFC", 35),
    ("Iraq", "IRQ", "AFC", 63),
    ("Uzbekistan", "UZB", "AFC", 62),

    # ── CAF (9) ───────────────────────────────────────────────────────
    ("Morocco", "MAR", "CAF", 18),
    ("Senegal", "SEN", "CAF", 20),
    ("Nigeria", "NGA", "CAF", 28),
    ("Egypt", "EGY", "CAF", 36),
    ("Cameroon", "CMR", "CAF", 46),
    ("South Africa", "RSA", "CAF", 59),
    ("Algeria", "ALG", "CAF", 32),
    ("Mali", "MLI", "CAF", 48),
    ("Ivory Coast", "CIV", "CAF", 39),

    # ── OFC (1) ───────────────────────────────────────────────────────
    ("New Zealand", "NZL", "OFC", 93),

    # ── Repechaje intercontinental (2) ────────────────────────────────
    ("Indonesia", "IDN", "AFC", 87),
    ("Trinidad and Tobago", "TRI", "CONCACAF", 103),
]

# Verificación de integridad
assert len(WORLD_CUP_2026_TEAMS) == 48, (
    f"Se esperan 48 equipos, se encontraron {len(WORLD_CUP_2026_TEAMS)}"
)


# ============================================================================
# COMPETICIONES HISTÓRICAS
# ============================================================================
# Formato: (competition_name, season, stage)
# Incluye las competiciones cuyos datos históricos alimentan el modelo.

COMPETITIONS: list[tuple[str, str, str]] = [
    # Mundiales
    ("FIFA World Cup", "2014", "Finals"),
    ("FIFA World Cup", "2018", "Finals"),
    ("FIFA World Cup", "2022", "Finals"),
    ("FIFA World Cup", "2026", "Finals"),

    # Eliminatorias mundialistas
    ("WCQ - CONMEBOL", "2022", "Qualifiers"),
    ("WCQ - CONMEBOL", "2026", "Qualifiers"),
    ("WCQ - UEFA", "2022", "Qualifiers"),
    ("WCQ - UEFA", "2026", "Qualifiers"),
    ("WCQ - CONCACAF", "2022", "Qualifiers"),
    ("WCQ - CONCACAF", "2026", "Qualifiers"),
    ("WCQ - AFC", "2022", "Qualifiers"),
    ("WCQ - AFC", "2026", "Qualifiers"),
    ("WCQ - CAF", "2022", "Qualifiers"),
    ("WCQ - CAF", "2026", "Qualifiers"),

    # Torneos continentales
    ("Copa America", "2021", "Finals"),
    ("Copa America", "2024", "Finals"),
    ("UEFA Euro", "2020", "Finals"),
    ("UEFA Euro", "2024", "Finals"),
    ("UEFA Nations League", "2022-2023", "League"),
    ("UEFA Nations League", "2024-2025", "League"),
    ("Africa Cup of Nations", "2023", "Finals"),
    ("Africa Cup of Nations", "2025", "Finals"),
    ("AFC Asian Cup", "2023", "Finals"),
    ("CONCACAF Gold Cup", "2023", "Finals"),
    ("CONCACAF Nations League", "2024-2025", "League"),

    # Amistosos internacionales
    ("International Friendly", "2014-2026", "Friendly"),
]
