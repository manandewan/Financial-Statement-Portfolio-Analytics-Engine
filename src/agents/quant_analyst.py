import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger("QuantAnalyst")

class QuantAnalystAgent:
    """
    Agent 3: Quantitative Analyst
    Portfolio Optimizer relying on Modern Portfolio Theory (MPT), Value at Risk (VaR),
    Conditional VaR (CVaR), Beta/Alpha modeling, and mathematical optimization (SLSQP).
    """
    def __init__(self, risk_free_rate: float = 0.04):
        self.name = "Quantitative Analyst"
        self.risk_free_rate = risk_free_rate

    def calculate_asset_metrics(
        self, 
        prices_df: pd.DataFrame, 
        return_multiplier: float = 1.0,
        ml_return_forecasts: Dict[str, float] = None,
        use_ml_views: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate individual asset risk & return metrics: CAGR, Volatility, Max Drawdown,
        Sharpe Ratio, 95% Value at Risk (VaR), and 95% Conditional VaR (CVaR).
        """
        returns_df = prices_df.pct_change().dropna()
        tickers = list(prices_df.columns)

        if prices_df.empty or len(prices_df) < 2:
            num_years = 1.0
        else:
            num_years = max((prices_df.index[-1] - prices_df.index[0]).days / 365.25, 0.5)

        asset_metrics = {}
        mean_returns_dict = {}

        for ticker in tickers:
            prices = prices_df[ticker].dropna()
            if prices.empty:
                continue

            start_p = prices.iloc[0]
            end_p = prices.iloc[-1]
            cagr = ((end_p / (start_p + 1e-8)) ** (1.0 / num_years) - 1.0) * return_multiplier if start_p > 0 else 0.0

            if ticker in returns_df.columns and not returns_df[ticker].empty:
                daily_rets = returns_df[ticker]
                
                # Base annualized return
                if use_ml_views and ml_return_forecasts and ticker in ml_return_forecasts:
                    ann_return = float(ml_return_forecasts[ticker]) * return_multiplier
                else:
                    ann_return = float(daily_rets.mean() * 252 * return_multiplier)

                ann_vol = float(daily_rets.std() * np.sqrt(252))
                cum_rets = (1 + daily_rets).cumprod()
                peak = cum_rets.cummax()
                drawdown = (cum_rets - peak) / (peak + 1e-8)
                max_drawdown = float(drawdown.min())

                # 95% Historical Value at Risk (VaR) & Conditional VaR (CVaR)
                var_95 = float(np.percentile(daily_rets, 5) * np.sqrt(252))
                cvar_95 = float(daily_rets[daily_rets <= np.percentile(daily_rets, 5)].mean() * np.sqrt(252))
            else:
                ann_return = 0.0
                ann_vol = 1e-4
                max_drawdown = 0.0
                var_95 = 0.0
                cvar_95 = 0.0

            sharpe = (ann_return - self.risk_free_rate) / (ann_vol + 1e-8) if ann_vol > 0 else 0

            asset_metrics[ticker] = {
                'cagr': float(cagr),
                'annualized_return': float(ann_return),
                'annualized_volatility': float(ann_vol),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95)
            }
            mean_returns_dict[ticker] = ann_return

        cov_matrix = returns_df.cov() * 252 if not returns_df.empty else pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
        corr_matrix = returns_df.corr() if not returns_df.empty else pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
        mean_returns_series = pd.Series(mean_returns_dict)

        return {
            'asset_metrics': asset_metrics,
            'mean_returns': mean_returns_series,
            'cov_matrix': cov_matrix,
            'corr_matrix': corr_matrix,
            'returns_df': returns_df
        }

    def _portfolio_performance(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute portfolio return, volatility, and Sharpe ratio.
        """
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.maximum(np.dot(weights.T, np.dot(cov_matrix, weights)), 1e-8))
        sharpe = (port_return - self.risk_free_rate) / (port_vol + 1e-8)
        return float(port_return), float(port_vol), float(sharpe)

    def _neg_sharpe_ratio(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        return -self._portfolio_performance(weights, mean_returns, cov_matrix)[2]

    def _portfolio_volatility(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        return self._portfolio_performance(weights, mean_returns, cov_matrix)[1]

    def optimize_portfolio(
        self, 
        raw_data: Dict[str, Any], 
        risk_free_rate: float = None,
        return_multiplier: float = 1.0,
        ml_return_forecasts: Dict[str, float] = None,
        use_ml_views: bool = False
    ) -> Dict[str, Any]:
        """
        Compute optimal portfolio allocations, Efficient Frontier, and dynamic parameter adjustments.
        Supports ML-enhanced Black-Litterman expected return views.
        """
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate

        prices_df = raw_data.get('prices', pd.DataFrame())
        if prices_df.empty or prices_df.shape[1] < 1:
            raise ValueError("Prices DataFrame is empty or missing columns.")

        tickers = list(prices_df.columns)
        num_assets = len(tickers)

        metrics_res = self.calculate_asset_metrics(
            prices_df, 
            return_multiplier=return_multiplier,
            ml_return_forecasts=ml_return_forecasts,
            use_ml_views=use_ml_views
        )
        mean_returns = metrics_res['mean_returns'].values
        cov_matrix = metrics_res['cov_matrix'].values
        corr_matrix = metrics_res['corr_matrix']
        returns_df = metrics_res['returns_df']

        init_weights = np.array([1.0 / num_assets] * num_assets)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        if num_assets == 1:
            max_sharpe_weights = np.array([1.0])
            min_var_weights = np.array([1.0])
            max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = self._portfolio_performance(
                max_sharpe_weights, mean_returns, cov_matrix
            )
            min_var_ret, min_var_vol, min_var_sr = max_sharpe_ret, max_sharpe_vol, max_sharpe_sr
            target_returns = [float(mean_returns[0])]
            efficient_volatilities = [float(max_sharpe_vol)]
            efficient_weights = [[1.0]]
            mc_returns = [float(mean_returns[0])]
            mc_volatilities = [float(max_sharpe_vol)]
            mc_sharpe = [float(max_sharpe_sr)]
        else:
            # 1. Max Sharpe Ratio Optimization (SLSQP)
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

            # 2. Minimum Variance Optimization (SLSQP)
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
            min_r = float(min(mean_returns))
            max_r = float(max(mean_returns))
            if abs(min_r - max_r) < 1e-5:
                target_returns = np.array([min_r])
            else:
                target_returns = np.linspace(min_r, max_r, 50)

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
                    efficient_volatilities.append(float(res.fun))
                    efficient_weights.append(res.x.tolist())
                else:
                    efficient_volatilities.append(np.nan)
                    efficient_weights.append([0.0]*num_assets)

            # 4. Monte Carlo Simulation (2,500 iterations)
            num_simulations = 2500
            mc_returns = np.zeros(num_simulations)
            mc_volatilities = np.zeros(num_simulations)
            mc_sharpe = np.zeros(num_simulations)

            np.random.seed(42)
            for i in range(num_simulations):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                r, v, s = self._portfolio_performance(w, mean_returns, cov_matrix)
                mc_returns[i] = r
                mc_volatilities[i] = v
                mc_sharpe[i] = s

            target_returns = target_returns.tolist()
            mc_returns = mc_returns.tolist()
            mc_volatilities = mc_volatilities.tolist()
            mc_sharpe = mc_sharpe.tolist()

        # Calculate Max Sharpe Portfolio VaR & CVaR
        ms_daily_returns = returns_df.dot(max_sharpe_weights) if not returns_df.empty else pd.Series([0.0])
        ms_var95 = float(np.percentile(ms_daily_returns, 5) * np.sqrt(252))
        ms_cvar95 = float(ms_daily_returns[ms_daily_returns <= np.percentile(ms_daily_returns, 5)].mean() * np.sqrt(252)) if len(ms_daily_returns) > 5 else ms_var95

        max_sharpe_dict = {
            'weights': {tickers[i]: float(max_sharpe_weights[i]) for i in range(num_assets)},
            'expected_return': float(max_sharpe_ret),
            'volatility': float(max_sharpe_vol),
            'sharpe_ratio': float(max_sharpe_sr),
            'var_95': float(ms_var95),
            'cvar_95': float(ms_cvar95)
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
            'mean_returns': mean_returns.tolist(),
            'returns_df': returns_df,
            'max_sharpe_portfolio': max_sharpe_dict,
            'min_variance_portfolio': min_var_dict,
            'efficient_frontier': {
                'target_returns': target_returns,
                'volatilities': efficient_volatilities,
                'weights': efficient_weights
            },
            'monte_carlo': {
                'returns': mc_returns,
                'volatilities': mc_volatilities,
                'sharpe_ratios': mc_sharpe
            },
            'correlation_matrix': corr_matrix.to_dict(),
            'risk_free_rate': self.risk_free_rate,
            'return_multiplier': return_multiplier,
            'use_ml_views': use_ml_views
        }
