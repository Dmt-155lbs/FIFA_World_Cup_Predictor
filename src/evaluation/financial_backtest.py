import pandas as pd
import structlog
import matplotlib.pyplot as plt

logger = structlog.get_logger(__name__)

class FinancialBacktester:
    """
    Simula rentabilidad cruzando predicciones del modelo
    contra cuotas históricas reales de casas de apuestas.
    """

    def _calculate_max_drawdown(self, bankroll_history: list[float]) -> float:
        """Calcula el máximo drawdown relativo (porcentaje)."""
        if not bankroll_history:
            return 0.0
        peak = bankroll_history[0]
        max_dd = 0.0
        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100.0

    def flat_stake_roi(self, predictions: list[dict], odds: list[dict], actuals: list[str], stake: float = 1.0, ev_threshold: float = 0.05) -> dict:
        """
        Estrategia Flat Stake.
        predictions: [{'home': 0.45, 'draw': 0.25, 'away': 0.30}, ...]
        odds: [{'home': 2.10, 'draw': 3.40, 'away': 3.50}, ...]
        actuals: ['home', 'away', 'draw', ...]
        """
        total_bets = 0
        total_profit = 0.0
        bets_log = []
        bankroll = 0.0
        bankroll_history = [0.0]

        for pred, odd, actual in zip(predictions, odds, actuals):
            for outcome in ['home', 'draw', 'away']:
                # Calcular expected value
                ev = (pred[outcome] * odd[outcome]) - 1

                if ev > ev_threshold:
                    total_bets += 1
                    won = (outcome == actual)
                    profit = (odd[outcome] - 1) * stake if won else -stake
                    total_profit += profit
                    bankroll += profit
                    bankroll_history.append(bankroll)

                    bets_log.append({
                        'outcome_bet': outcome,
                        'prob_model': pred[outcome],
                        'odds': odd[outcome],
                        'ev': ev,
                        'won': won,
                        'profit': profit
                    })

        roi = (total_profit / (total_bets * stake)) if total_bets > 0 else 0.0
        win_rate = sum(b['won'] for b in bets_log) / max(len(bets_log), 1)

        logger.info("Flat Stake ROI calculado.", roi=roi, total_bets=total_bets)
        return {
            'total_bets': total_bets,
            'total_profit': total_profit,
            'roi_pct': roi * 100,
            'win_rate': win_rate,
            'bets_log': bets_log,
            'bankroll_history': bankroll_history
        }

    def kelly_criterion_roi(self, predictions: list[dict], odds: list[dict], actuals: list[str], 
                            fraction: float = 0.25, initial_bankroll: float = 1000.0) -> dict:
        """
        Estrategia Kelly Criterion (fraccionario).
        """
        current_bankroll = initial_bankroll
        bankroll_history = [initial_bankroll]
        total_bets = 0

        for pred, odd, actual in zip(predictions, odds, actuals):
            for outcome in ['home', 'draw', 'away']:
                p = pred[outcome]
                b = odd[outcome] - 1  # Ganancia neta por unidad apostada
                q = 1 - p

                kelly_full = (b * p - q) / b if b > 0 else 0
                kelly_frac = max(0, kelly_full * fraction)

                if kelly_frac > 0.01:  # Mínimo 1% del bankroll
                    bet_size = current_bankroll * kelly_frac
                    won = (outcome == actual)
                    total_bets += 1
                    
                    if won:
                        current_bankroll += bet_size * b
                    else:
                        current_bankroll -= bet_size
                        
                    bankroll_history.append(current_bankroll)

        roi = ((current_bankroll - initial_bankroll) / initial_bankroll) * 100
        max_dd = self._calculate_max_drawdown(bankroll_history)

        logger.info("Kelly Criterion ROI calculado.", roi=roi, total_bets=total_bets)
        return {
            'final_bankroll': current_bankroll,
            'roi_pct': roi,
            'max_drawdown': max_dd,
            'total_bets': total_bets,
            'bankroll_history': bankroll_history
        }

    def plot_roi_curves(self, flat_history: list[float], kelly_history: list[float], save_path: str = "roi_curve.png"):
        """
        Plotea las curvas acumuladas de P&L de las dos estrategias.
        """
        # Limpiar figura actual por si acaso
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Bets')
        ax1.set_ylabel('Flat Stake P&L', color=color)
        ax1.plot(flat_history, color=color, label='Flat Stake')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('Kelly Criterion Bankroll', color=color)  
        ax2.plot(kelly_history, color=color, label='Kelly Criterion (Frac)')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Backtesting Financiero: P&L Curves')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("ROI curves saved.", path=save_path)
