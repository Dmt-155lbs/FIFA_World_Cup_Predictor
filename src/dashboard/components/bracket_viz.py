import plotly.graph_objects as go

# Orden canónico de las rondas eliminatorias (claves de round_advance_probs)
_ROUND_ORDER = [
    "Ronda de 32",
    "Octavos de Final",
    "Cuartos de Final",
    "Semifinal",
    "Final",
]


def plot_bracket(round_advance_probs, top_n: int = 12):
    """
    Visualiza la probabilidad de avance por ronda para los principales equipos.

    Parameters
    ----------
    round_advance_probs : dict[str, dict[str, float]] | None
        ``{ronda: {equipo: probabilidad}}`` proveniente de la simulación
        Monte Carlo. Si es ``None`` o vacío, se muestra un aviso.
    top_n : int
        Número de equipos a mostrar (ordenados por prob. de llegar a la Final).
    """
    if not round_advance_probs:
        fig = go.Figure()
        fig.add_annotation(
            text="Sin datos de simulación disponibles",
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(height=400)
        return fig

    rounds = [r for r in _ROUND_ORDER if r in round_advance_probs]
    # Rankear equipos por su probabilidad de llegar a la ronda más avanzada
    deepest = rounds[-1] if rounds else None
    ranking_source = round_advance_probs.get(deepest, {})
    top_teams = [
        t for t, _ in sorted(
            ranking_source.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
    ]

    fig = go.Figure()
    for ronda in rounds:
        probs = round_advance_probs.get(ronda, {})
        fig.add_trace(
            go.Bar(
                name=ronda,
                x=top_teams,
                y=[probs.get(t, 0.0) for t in top_teams],
                text=[f"{probs.get(t, 0.0):.0%}" for t in top_teams],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Probabilidad de avance por ronda (Top equipos)",
        yaxis_title="Probabilidad",
        yaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Ronda",
        height=480,
    )
    return fig

def plot_champions_distribution(champions_probs):
    """
    Gráfico de barras con los favoritos al título.
    champions_probs: dict {equipo: prob}
    """
    # Ordenar y tomar el Top 10
    sorted_probs = sorted(champions_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    teams = [x[0] for x in sorted_probs]
    probs = [x[1] for x in sorted_probs]
    
    fig = go.Figure(go.Bar(
        x=teams,
        y=probs,
        marker_color='#e94560',
        text=[f"{p:.1%}" for p in probs],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Probabilidad de Ganar el Mundial (Top 10)",
        yaxis_title="Probabilidad",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat='.0%'
    )
    return fig
