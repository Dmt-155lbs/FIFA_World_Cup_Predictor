import plotly.express as px
import numpy as np

def plot_poisson_heatmap(matrix, home_team, away_team, max_goals=5):
    """
    Renderiza un heatmap interactivo usando Plotly para la matriz de Poisson.
    Muestra P(Local = x, Visitante = y).
    """
    # Recortar a max_goals para la visualización
    z = matrix[:max_goals+1, :max_goals+1]
    
    # Crear etiquetas para ejes
    x_labels = [f"{i}" for i in range(max_goals+1)]
    y_labels = [f"{i}" for i in range(max_goals+1)]
    
    fig = px.imshow(
        z,
        labels=dict(x=f"Goles {away_team} (Visitante)", y=f"Goles {home_team} (Local)", color="Probabilidad"),
        x=x_labels,
        y=y_labels,
        color_continuous_scale="Blues",
        text_auto=".1%",
        aspect="auto"
    )
    
    fig.update_layout(
        title=f"Matriz Poisson: {home_team} vs {away_team}",
        xaxis_title=f"Goles {away_team}",
        yaxis_title=f"Goles {home_team}",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Invertir eje y para que 0 esté abajo
    fig.update_yaxes(autorange="reversed")
    
    return fig
