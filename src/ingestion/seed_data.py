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
# 48 EQUIPOS CLASIFICADOS — MUNDIAL FIFA 2026 (LISTA OFICIAL)
# ============================================================================
# Formato: (team_name, fifa_code, confederation, fifa_ranking)
# Esta es la composición OFICIAL de clasificados (no el armado provisional por
# bombos). Refleja el sorteo final por grupos A–L de bracket_2026.yaml.
# Selecciones que NO clasificaron y se retiraron del listado anterior:
#   Italia, Dinamarca, Serbia, Gales, Ucrania (UEFA); Jamaica, Honduras
#   (CONCACAF); Nigeria, Camerún, Malí (CAF); Indonesia, Trinidad y Tobago.
# Reemplazadas por los clasificados reales (ver más abajo).
# El ranking FIFA es aproximado a junio 2026 (solo ordena pots/listas).

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
    ("Croatia", "CRO", "UEFA", 13),
    ("Switzerland", "SUI", "UEFA", 19),
    ("Austria", "AUT", "UEFA", 22),
    ("Turkey", "TUR", "UEFA", 26),
    ("Norway", "NOR", "UEFA", 32),          # nuevo: clasificó (grupo con Italia)
    ("Sweden", "SWE", "UEFA", 38),          # nuevo
    ("Czech Republic", "CZE", "UEFA", 40),  # nuevo
    ("Scotland", "SCO", "UEFA", 45),        # nuevo
    ("Bosnia and Herzegovina", "BIH", "UEFA", 74),  # nuevo

    # ── CONCACAF (6 — incluye 3 anfitriones) ──────────────────────────
    ("United States", "USA", "CONCACAF", 14),
    ("Mexico", "MEX", "CONCACAF", 15),
    ("Canada", "CAN", "CONCACAF", 41),
    ("Panama", "PAN", "CONCACAF", 44),
    ("Haiti", "HAI", "CONCACAF", 83),       # nuevo
    ("Curacao", "CUW", "CONCACAF", 90),     # nuevo

    # ── AFC (9) ───────────────────────────────────────────────────────
    ("Japan", "JPN", "AFC", 16),
    ("Iran", "IRN", "AFC", 20),
    ("South Korea", "KOR", "AFC", 21),
    ("Australia", "AUS", "AFC", 24),
    ("Qatar", "QAT", "AFC", 35),
    ("Saudi Arabia", "KSA", "AFC", 56),
    ("Uzbekistan", "UZB", "AFC", 62),
    ("Iraq", "IRQ", "AFC", 63),
    ("Jordan", "JOR", "AFC", 64),           # nuevo

    # ── CAF (10) ──────────────────────────────────────────────────────
    ("Morocco", "MAR", "CAF", 18),
    ("Senegal", "SEN", "CAF", 20),
    ("Algeria", "ALG", "CAF", 33),
    ("Egypt", "EGY", "CAF", 36),
    ("Ivory Coast", "CIV", "CAF", 39),
    ("Tunisia", "TUN", "CAF", 41),          # nuevo
    ("DR Congo", "COD", "CAF", 57),         # nuevo
    ("South Africa", "RSA", "CAF", 59),
    ("Cape Verde", "CPV", "CAF", 70),       # nuevo
    ("Ghana", "GHA", "CAF", 73),            # nuevo

    # ── OFC (1) ───────────────────────────────────────────────────────
    ("New Zealand", "NZL", "OFC", 93),
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
