import plotly.express as px
import pandas as pd
import numpy as np

def plot_roi_curves(backtest_results):
    """
    Renderiza las curvas de Profit & Loss del backtest.
    """
    # Simulación de datos para demostración si backtest_results es None
    if backtest_results is None:
        dates = pd.date_range(start="2022-11-20", end="2022-12-18")
        df = pd.DataFrame({
            'Fecha': dates,
            'Flat Stake': [1000 + i*5 + np.random.normal(0, 10) for i in range(len(dates))],
            'Kelly Criterion': [1000 + i*8 + np.random.normal(0, 15) for i in range(len(dates))],
            'Baseline (Naive)': [1000 - i*2 for i in range(len(dates))]
        })
    else:
        df = pd.DataFrame(backtest_results)
        
    # Melt para plotly
    df_melt = df.melt(id_vars=['Fecha'], var_name='Estrategia', value_name='Bankroll ($)')
    
    fig = px.line(
        df_melt,
        x='Fecha',
        y='Bankroll ($)',
        color='Estrategia',
        title='Curvas de Profit & Loss Acumuladas',
        color_discrete_map={
            'Flat Stake': '#00b4d8',
            'Kelly Criterion': '#e94560',
            'Baseline (Naive)': '#8d99ae'
        }
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    
    return fig
