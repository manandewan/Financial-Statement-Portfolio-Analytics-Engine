import unittest
import pandas as pd
import numpy as np

from src.agents.data_architect import DataArchitectAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.quant_analyst import QuantAnalystAgent
from src.agents.ml_predictive_analyst import MLPredictiveAnalystAgent
from src.agents.coordinator import AgentSystemCoordinator

class TestAgents(unittest.TestCase):

    def setUp(self):
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

        self.sample_statements = {
            'AAPL': {
                'income_statement': pd.DataFrame({'2024': {'Net Income': 100000, 'Total Revenue': 400000}}),
                'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 500000, 'Total Debt': 200000, 'Current Assets': 150000, 'Current Liabilities': 100000}}),
                'cash_flow': pd.DataFrame({'2024': {'Free Cash Flow': 90000, 'Operating Cash Flow': 110000, 'Capital Expenditure': -20000}}),
                'info': {'marketCap': 2000000, 'sector': 'Technology'}
            },
            'MSFT': {
                'income_statement': pd.DataFrame({'2024': {'Net Income': 80000, 'Total Revenue': 300000}}),
                'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 400000, 'Total Debt': 300000, 'Current Assets': 120000, 'Current Liabilities': 80000}}),
                'cash_flow': pd.DataFrame({'2024': {'Free Cash Flow': 70000, 'Operating Cash Flow': 90000, 'Capital Expenditure': -20000}}),
                'info': {'marketCap': 1800000, 'sector': 'Technology'}
            },
            'GOOGL': {
                'income_statement': pd.DataFrame({'2024': {'Net Income': 70000, 'Total Revenue': 280000}}),
                'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 350000, 'Total Debt': 100000, 'Current Assets': 180000, 'Current Liabilities': 90000}}),
                'cash_flow': pd.DataFrame({'2024': {'Free Cash Flow': 60000, 'Operating Cash Flow': 75000, 'Capital Expenditure': -15000}}),
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
        self.assertAlmostEqual(aapl_m['debt_to_equity'], 0.4, places=2)
        self.assertAlmostEqual(aapl_m['current_ratio'], 1.5, places=2)
        self.assertAlmostEqual(aapl_m['return_on_equity'], 0.20, places=2)

    def test_ml_predictive_analyst_agent(self):
        ml_agent = MLPredictiveAnalystAgent(model_type="random_forest")
        res = ml_agent.predict(self.raw_data)
        self.assertIn('ml_results', res)
        self.assertIn('latest_features', res)
        self.assertIn('AAPL', res['ml_results'])

    def test_quant_analyst_agent_and_var(self):
        quant = QuantAnalystAgent(risk_free_rate=0.04)
        opt_res = quant.optimize_portfolio(self.raw_data, return_multiplier=1.2)

        self.assertIn('max_sharpe_portfolio', opt_res)
        self.assertIn('min_variance_portfolio', opt_res)
        self.assertIn('var_95', opt_res['max_sharpe_portfolio'])
        self.assertIn('cvar_95', opt_res['max_sharpe_portfolio'])

        weights = opt_res['max_sharpe_portfolio']['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)

    def test_quant_with_ml_views(self):
        ml_views = {'AAPL': 0.25, 'MSFT': 0.15, 'GOOGL': 0.10}
        quant = QuantAnalystAgent(risk_free_rate=0.04)
        opt_res = quant.optimize_portfolio(self.raw_data, ml_return_forecasts=ml_views, use_ml_views=True)
        self.assertTrue(opt_res['use_ml_views'])
        self.assertIn('max_sharpe_portfolio', opt_res)

    def test_single_asset_quant_edge_case(self):
        single_data = {
            'prices': self.sample_prices[['AAPL']],
            'tickers': ['AAPL'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'statements': {'AAPL': self.sample_statements['AAPL']}
        }
        quant = QuantAnalystAgent(risk_free_rate=0.04)
        opt_res = quant.optimize_portfolio(single_data)
        self.assertEqual(opt_res['max_sharpe_portfolio']['weights']['AAPL'], 1.0)

    def test_coordinator_full_pipeline(self):
        coordinator = AgentSystemCoordinator()
        coordinator.data_architect.fetch_data = lambda tickers, start_date, end_date: self.raw_data
        bundle = coordinator.run_pipeline(['AAPL', 'MSFT', 'GOOGL'], risk_free_rate=0.04, use_ml_views=True)

        self.assertIn('fundamental', bundle)
        self.assertIn('ml_predictive', bundle)
        self.assertIn('quant', bundle)
        self.assertIn('ai_report', bundle)

if __name__ == '__main__':
    unittest.main()
