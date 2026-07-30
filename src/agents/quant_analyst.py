import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger("QuantAnalyst")

class QuantAnalystAgent:
    """
    Agent 3: Quantitative Analyst
    Portfolio Optimizer relying on Modern Portfolio Theory (MPT) and mathematical optimization
    to calculate risk/return metrics, efficient frontier, and optimal asset allocations.
    """
    def __init__(self, risk_free_rate: float = 0.04):
        self.name = "Quantitative Analyst"
        self.risk_free_rate = risk_free_rate

    def calculate_asset_metrics(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate individual asset risk & return metrics: CAGR, Volatility, Max Drawdown, Sharpe.
        """
        returns_df = prices_df.pct_change().dropna()
        tickers = list(prices_df.columns)

        num_years = max((prices_df.index[-1] - prices_df.index[0]).days / 365.25, 0.5)

        asset_metrics = {}
        for ticker in tickers:
            prices = prices_df[ticker].dropna()
            if prices.empty:
                continue

            start_p = prices.iloc[0]
            end_p = prices.iloc[-1]
            cagr = (end_p / start_p) ** (1.0 / num_years) - 1.0

            daily_rets = returns_df[ticker]
            ann_return = daily_rets.mean() * 252
            ann_vol = daily_rets.std() * np.sqrt(252)

            sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0

            # Maximum Drawdown
            cum_rets = (1 + daily_rets).cumprod()
            peak = cum_rets.cummax()
            drawdown = (cum_rets - peak) / peak
            max_drawdown = float(drawdown.min())

            asset_metrics[ticker] = {
                'cagr': float(cagr),
                'annualized_return': float(ann_return),
                'annualized_volatility': float(ann_vol),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown)
            }

        cov_matrix = returns_df.cov() * 252
        corr_matrix = returns_df.corr()

        return {
            'asset_metrics': asset_metrics,
            'mean_returns': returns_df.mean() * 252,
            'cov_matrix': cov_matrix,
            'corr_matrix': corr_matrix,
            'returns_df': returns_df
        }

    def _portfolio_performance(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute portfolio return, volatility, and Sharpe ratio.
        """
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        return port_return, port_vol, sharpe

    def _neg_sharpe_ratio(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        return -self._portfolio_performance(weights, mean_returns, cov_matrix)[2]

    def _portfolio_volatility(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        return self._portfolio_performance(weights, mean_returns, cov_matrix)[1]

    def optimize_portfolio(self, raw_data: Dict[str, Any], risk_free_rate: float = None) -> Dict[str, Any]:
        """
        Compute optimal portfolio allocations, Efficient Frontier, and Monte Carlo portfolio simulation.
        """
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate

        prices_df = raw_data.get('prices', pd.DataFrame())
        if prices_df.empty or prices_df.shape[1] < 1:
            raise ValueError("Prices DataFrame is empty or missing columns.")

        tickers = list(prices_df.columns)
        num_assets = len(tickers)

        metrics_res = self.calculate_asset_metrics(prices_df)
        mean_returns = metrics_res['mean_returns'].values
        cov_matrix = metrics_res['cov_matrix'].values
        corr_matrix = metrics_res['corr_matrix']

        # Equal weight baseline
        init_weights = np.array([1.0 / num_assets] * num_assets)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        # 1. Max Sharpe Ratio Optimization
        opt_sharpe = minimize(
            self._neg_sharpe_ratio,
            init_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        max_sharpe_weights = opt_sharpe.x if opt_sharpe.success else init_weights
        max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = self._portfolio_performance(
            max_sharpe_weights, mean_returns, cov_matrix
        )

        # 2. Minimum Variance Optimization
        opt_min_var = minimize(
            self._portfolio_volatility,
            init_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        min_var_weights = opt_min_var.x if opt_min_var.success else init_weights
        min_var_ret, min_var_vol, min_var_sr = self._portfolio_performance(
            min_var_weights, mean_returns, cov_matrix
        )

        # 3. Efficient Frontier curve generation
        target_returns = np.linspace(min(mean_returns), max(mean_returns), 50)
        efficient_volatilities = []
        efficient_weights = []

        for target in target_returns:
            target_constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(mean_returns * w) - target}
            )
            res = minimize(
                self._portfolio_volatility,
                init_weights,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=target_constraints
            )
            if res.success:
                efficient_volatilities.append(res.fun)
                efficient_weights.append(res.x.tolist())
            else:
                efficient_volatilities.append(np.nan)
                efficient_weights.append([0.0]*num_assets)

        # 4. Monte Carlo Simulation (for background visual scatter plot)
        num_simulations = 2500
        mc_returns = np.zeros(num_simulations)
        mc_volatilities = np.zeros(num_simulations)
        mc_sharpe = np.zeros(num_simulations)
        mc_weights = np.zeros((num_simulations, num_assets))

        np.random.seed(42)
        for i in range(num_simulations):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            r, v, s = self._portfolio_performance(w, mean_returns, cov_matrix)
            mc_returns[i] = r
            mc_volatilities[i] = v
            mc_sharpe[i] = s
            mc_weights[i] = w

        # Build clean output structures
        max_sharpe_dict = {
            'weights': {tickers[i]: float(max_sharpe_weights[i]) for i in range(num_assets)},
            'expected_return': float(max_sharpe_ret),
            'volatility': float(max_sharpe_vol),
            'sharpe_ratio': float(max_sharpe_sr)
        }

        min_var_dict = {
            'weights': {tickers[i]: float(min_var_weights[i]) for i in range(num_assets)},
            'expected_return': float(min_var_ret),
            'volatility': float(min_var_vol),
            'sharpe_ratio': float(min_var_sr)
        }

        return {
            'tickers': tickers,
            'asset_metrics': metrics_res['asset_metrics'],
            'returns_df': metrics_res['returns_df'],
            'max_sharpe_portfolio': max_sharpe_dict,
            'min_variance_portfolio': min_var_dict,
            'efficient_frontier': {
                'target_returns': target_returns.tolist(),
                'volatilities': efficient_volatilities,
                'weights': efficient_weights
            },
            'monte_carlo': {
                'returns': mc_returns.tolist(),
                'volatilities': mc_volatilities.tolist(),
                'sharpe_ratios': mc_sharpe.tolist()
            },
            'correlation_matrix': corr_matrix.to_dict(),
            'risk_free_rate': self.risk_free_rate
        }
