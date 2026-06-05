import plotly.graph_objects as go

def plot_bracket(bracket_data):
    """
    Visualiza el bracket del torneo usando un gráfico tipo sankey o de árbol.
    Por simplicidad usaremos un diagrama de árbol básico simulado.
    """
    # Mock data si bracket_data es None
    # En producción usar graph_objects Sankey o un plot jerárquico.
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = 1,
        title = {"text": "Módulo Bracket en Desarrollo"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=400)
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
