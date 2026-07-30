import unittest
import pandas as pd
import numpy as np
import datetime

from src.agents.data_architect import DataArchitectAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.quant_analyst import QuantAnalystAgent
from src.agents.coordinator import AgentSystemCoordinator

class TestAgents(unittest.TestCase):

    def setUp(self):
        # Create synthetic price data for 3 assets over 250 days
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=250, freq="B")
        
        ret_a = np.random.normal(0.0008, 0.012, 250)
        ret_b = np.random.normal(0.0005, 0.015, 250)
        ret_c = np.random.normal(0.0003, 0.008, 250)

        price_a = 100 * np.cumprod(1 + ret_a)
        price_b = 50 * np.cumprod(1 + ret_b)
        price_c = 200 * np.cumprod(1 + ret_c)

        self.sample_prices = pd.DataFrame({
            'AAPL': price_a,
            'MSFT': price_b,
            'GOOGL': price_c
        }, index=dates)

        # Synthetic financial statement data
        self.sample_statements = {
            'AAPL': {
                'income_statement': pd.DataFrame({'2024': {'Net Income': 100000, 'Total Revenue': 400000}}),
                'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 500000, 'Total Debt': 200000, 'Current Assets': 150000, 'Current Liabilities': 100000}}),
                'cash_flow': pd.DataFrame({'2024': {'Free Cash Flow': 90000}}),
                'info': {'marketCap': 2000000, 'sector': 'Technology'}
            },
            'MSFT': {
                'income_statement': pd.DataFrame({'2024': {'Net Income': 80000, 'Total Revenue': 300000}}),
                'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 400000, 'Total Debt': 300000, 'Current Assets': 120000, 'Current Liabilities': 80000}}),
                'cash_flow': pd.DataFrame({'2024': {'Free Cash Flow': 70000}}),
                'info': {'marketCap': 1800000, 'sector': 'Technology'}
            },
            'GOOGL': {
                'income_statement': pd.DataFrame({'2024': {'Net Income': 70000, 'Total Revenue': 280000}}),
                'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 350000, 'Total Debt': 100000, 'Current Assets': 180000, 'Current Liabilities': 90000}}),
                'cash_flow': pd.DataFrame({'2024': {'Free Cash Flow': 60000}}),
                'info': {'marketCap': 1500000, 'sector': 'Communication Services'}
            }
        }

        self.raw_data = {
            'prices': self.sample_prices,
            'tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'statements': self.sample_statements
        }

    def test_fundamental_analyst_agent(self):
        analyst = FundamentalAnalystAgent()
        res = analyst.analyze(self.raw_data)
        metrics = res['metrics']

        self.assertIn('AAPL', metrics)
        aapl_m = metrics['AAPL']

        # D/E = 200,000 / 500,000 = 0.4
        self.assertAlmostEqual(aapl_m['debt_to_equity'], 0.4, places=2)
        # Current Ratio = 150,000 / 100,000 = 1.5
        self.assertAlmostEqual(aapl_m['current_ratio'], 1.5, places=2)
        # ROE = 100,000 / 500,000 = 0.20
        self.assertAlmostEqual(aapl_m['return_on_equity'], 0.20, places=2)
        # FCF Yield = 90,000 / 2,000,000 = 0.045
        self.assertAlmostEqual(aapl_m['free_cash_flow_yield'], 0.045, places=3)

    def test_quant_analyst_agent(self):
        quant = QuantAnalystAgent(risk_free_rate=0.04)
        opt_res = quant.optimize_portfolio(self.raw_data)

        self.assertIn('max_sharpe_portfolio', opt_res)
        self.assertIn('min_variance_portfolio', opt_res)
        self.assertIn('efficient_frontier', opt_res)

        weights = opt_res['max_sharpe_portfolio']['weights']
        sum_weights = sum(weights.values())
        self.assertAlmostEqual(sum_weights, 1.0, places=4)

        for t, w in weights.items():
            self.assertGreaterEqual(w, -1e-5)

    def test_coordinator(self):
        coordinator = AgentSystemCoordinator()
        # Mocking data architect for quick offline test
        coordinator.data_architect.fetch_data = lambda tickers, start_date, end_date: self.raw_data
        bundle = coordinator.run_pipeline(['AAPL', 'MSFT', 'GOOGL'])

        self.assertIn('fundamental', bundle)
        self.assertIn('quant', bundle)
        self.assertEqual(len(bundle['tickers']), 3)

if __name__ == '__main__':
    unittest.main()
