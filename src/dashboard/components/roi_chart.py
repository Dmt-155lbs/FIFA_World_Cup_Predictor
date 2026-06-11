import plotly.express as px
import pandas as pd

def plot_roi_curves(flat_history: list[float], kelly_history: list[float]):
    """
    Renderiza las curvas de Profit & Loss acumuladas del backtest.

    Parameters
    ----------
    flat_history : list[float]
        Evolución del bankroll con la estrategia Flat Stake
        (``FinancialBacktester.flat_stake_roi(...)['bankroll_history']``).
    kelly_history : list[float]
        Evolución del bankroll con Kelly fraccionario
        (``FinancialBacktester.kelly_criterion_roi(...)['bankroll_history']``).
    """
    records = []
    for i, v in enumerate(flat_history):
        records.append({"Apuesta": i, "Bankroll": v, "Estrategia": "Flat Stake"})
    for i, v in enumerate(kelly_history):
        records.append({"Apuesta": i, "Bankroll": v, "Estrategia": "Kelly Criterion"})
    df = pd.DataFrame(records)

    fig = px.line(
        df,
        x="Apuesta",
        y="Bankroll",
        color="Estrategia",
        title="Curvas de Profit & Loss Acumuladas",
        color_discrete_map={
            "Flat Stake": "#00b4d8",
            "Kelly Criterion": "#e94560",
        },
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )

    return fig
