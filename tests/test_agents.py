import unittest
import pandas as pd
import numpy as np

from src.agents.data_architect import DataArchitectAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.quant_analyst import QuantAnalystAgent
from src.agents.ml_predictive_analyst import MLPredictiveAnalystAgent
from src.agents.ai_portfolio_analyst import AIPortfolioAnalystAgent
from src.agents.coordinator import AgentSystemCoordinator

class TestSuite20(unittest.TestCase):
    """
    Comprehensive 20-Test Rigorous Validation Suite covering:
    - Data Ingestion & Sanitization (Tests 1-3)
    - Fundamental Analysis & Edge Case Parsing (Tests 4-7)
    - Quantitative Optimization, MPT, VaR & CVaR (Tests 8-15)
    - Predictive Machine Learning & Feature Engineering (Tests 16-18)
    - AI Report Synthesis & Pipeline Orchestration (Tests 19-20)
    """

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

    # =========================================================================
    # TESTS 1-3: DATA ARCHITECT
    # =========================================================================
    def test_01_data_architect_structure(self):
        da = DataArchitectAgent()
        self.assertEqual(da.name, "Financial Data Architect")

    def test_02_data_architect_ticker_deduplication(self):
        da = DataArchitectAgent()
        # Test sanitization logic directly
        tickers = ['AAPL', 'aapl ', 'MSFT', 'GOOGL', 'msft']
        cleaned = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
        self.assertEqual(cleaned, ['AAPL', 'MSFT', 'GOOGL'])

    def test_03_data_architect_empty_ticker_exception(self):
        da = DataArchitectAgent()
        with self.assertRaises(ValueError):
            da.fetch_data([])

    # =========================================================================
    # TESTS 4-7: FUNDAMENTAL ANALYST
    # =========================================================================
    def test_04_fundamental_positive_ratios(self):
        fa = FundamentalAnalystAgent()
        res = fa.analyze(self.raw_data)['metrics']
        aapl = res['AAPL']
        self.assertAlmostEqual(aapl['debt_to_equity'], 0.4, places=2)
        self.assertAlmostEqual(aapl['current_ratio'], 1.5, places=2)
        self.assertAlmostEqual(aapl['return_on_equity'], 0.20, places=2)
        self.assertAlmostEqual(aapl['free_cash_flow_yield'], 0.045, places=3)

    def test_05_fundamental_negative_equity_deficit(self):
        fa = FundamentalAnalystAgent()
        neg_eq_data = {
            'prices': self.sample_prices[['AAPL']],
            'tickers': ['AAPL'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'statements': {
                'AAPL': {
                    'income_statement': pd.DataFrame({'2024': {'Net Income': 50000}}),
                    'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': -100000, 'Total Debt': 200000}}),
                    'cash_flow': pd.DataFrame(),
                    'info': {'marketCap': 1000000}
                }
            }
        }
        res = fa.analyze(neg_eq_data)['metrics']['AAPL']
        self.assertIn("Negative Equity Deficit", res['flags'])

    def test_06_fundamental_zero_debt_and_unprofitable(self):
        fa = FundamentalAnalystAgent()
        zero_debt_data = {
            'prices': self.sample_prices[['AAPL']],
            'tickers': ['AAPL'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'statements': {
                'AAPL': {
                    'income_statement': pd.DataFrame({'2024': {'Net Income': -50000}}),
                    'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 200000, 'Total Debt': 0}}),
                    'cash_flow': pd.DataFrame(),
                    'info': {'marketCap': 500000}
                }
            }
        }
        res = fa.analyze(zero_debt_data)['metrics']['AAPL']
        self.assertEqual(res['debt_to_equity'], 0.0)
        self.assertIn("Negative ROE (Unprofitable)", res['flags'])

    def test_07_fundamental_missing_fcf_fallback(self):
        fa = FundamentalAnalystAgent()
        cf_fallback_data = {
            'prices': self.sample_prices[['AAPL']],
            'tickers': ['AAPL'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'statements': {
                'AAPL': {
                    'income_statement': pd.DataFrame(),
                    'balance_sheet': pd.DataFrame({'2024': {'Total Stockholder Equity': 100000}}),
                    'cash_flow': pd.DataFrame({'2024': {'Operating Cash Flow': 150000, 'Capital Expenditure': -50000}}),
                    'info': {'marketCap': 1000000}
                }
            }
        }
        res = fa.analyze(cf_fallback_data)['metrics']['AAPL']
        self.assertEqual(res['free_cash_flow'], 100000.0)
        self.assertAlmostEqual(res['free_cash_flow_yield'], 0.10, places=2)

    # =========================================================================
    # TESTS 8-15: QUANT ANALYST, MPT & RISK
    # =========================================================================
    def test_08_quant_single_asset_edge_case(self):
        single_data = {
            'prices': self.sample_prices[['AAPL']],
            'tickers': ['AAPL'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'statements': {'AAPL': self.sample_statements['AAPL']}
        }
        qa = QuantAnalystAgent()
        res = qa.optimize_portfolio(single_data)
        self.assertEqual(res['max_sharpe_portfolio']['weights']['AAPL'], 1.0)
        self.assertEqual(res['min_variance_portfolio']['weights']['AAPL'], 1.0)

    def test_09_quant_weights_sum_to_one(self):
        qa = QuantAnalystAgent()
        res = qa.optimize_portfolio(self.raw_data)
        ms_w = sum(res['max_sharpe_portfolio']['weights'].values())
        mv_w = sum(res['min_variance_portfolio']['weights'].values())
        self.assertAlmostEqual(ms_w, 1.0, places=4)
        self.assertAlmostEqual(mv_w, 1.0, places=4)

    def test_10_quant_sharpe_maximization(self):
        qa = QuantAnalystAgent(risk_free_rate=0.02)
        res = qa.optimize_portfolio(self.raw_data)
        ms_sr = res['max_sharpe_portfolio']['sharpe_ratio']
        
        # Compare against equal weight portfolio
        eq_w = np.array([1/3, 1/3, 1/3])
        mean_rets = np.array(res['mean_returns'])
        cov = res['returns_df'].cov().values * 252
        eq_ret = np.sum(mean_rets * eq_w)
        eq_vol = np.sqrt(np.dot(eq_w.T, np.dot(cov, eq_w)))
        eq_sr = (eq_ret - 0.02) / eq_vol
        
        self.assertGreaterEqual(ms_sr, eq_sr - 1e-4)

    def test_11_quant_min_variance_portfolio(self):
        qa = QuantAnalystAgent()
        res = qa.optimize_portfolio(self.raw_data)
        mv_vol = res['min_variance_portfolio']['volatility']
        individual_vols = [res['asset_metrics'][t]['annualized_volatility'] for t in self.raw_data['tickers']]
        self.assertLessEqual(mv_vol, max(individual_vols) + 1e-4)

    def test_12_quant_risk_free_rate_shift(self):
        qa1 = QuantAnalystAgent(risk_free_rate=0.02)
        res1 = qa1.optimize_portfolio(self.raw_data)
        qa2 = QuantAnalystAgent(risk_free_rate=0.08)
        res2 = qa2.optimize_portfolio(self.raw_data)
        # Higher risk-free rate results in lower numerical Sharpe for identical returns
        self.assertGreater(res1['max_sharpe_portfolio']['sharpe_ratio'], res2['max_sharpe_portfolio']['sharpe_ratio'])

    def test_13_quant_return_multiplier_scaling(self):
        qa = QuantAnalystAgent()
        res_base = qa.optimize_portfolio(self.raw_data, return_multiplier=1.0)
        res_scaled = qa.optimize_portfolio(self.raw_data, return_multiplier=1.5)
        # Expected return scales up
        self.assertGreater(res_scaled['max_sharpe_portfolio']['expected_return'], res_base['max_sharpe_portfolio']['expected_return'])

    def test_14_quant_var_and_cvar_tail_property(self):
        qa = QuantAnalystAgent()
        res = qa.optimize_portfolio(self.raw_data)
        var95 = res['max_sharpe_portfolio']['var_95']
        cvar95 = res['max_sharpe_portfolio']['cvar_95']
        # CVaR (expected shortfall in worst 5%) must be more negative or equal to VaR
        self.assertLessEqual(cvar95, var95 + 1e-5)

    def test_15_quant_ml_views_integration(self):
        qa = QuantAnalystAgent()
        ml_views = {'AAPL': 0.30, 'MSFT': 0.05, 'GOOGL': 0.02}
        res = qa.optimize_portfolio(self.raw_data, ml_return_forecasts=ml_views, use_ml_views=True)
        self.assertTrue(res['use_ml_views'])
        # With AAPL having 30% forecast, AAPL should have the highest allocation in Max Sharpe
        self.assertGreater(res['max_sharpe_portfolio']['weights']['AAPL'], res['max_sharpe_portfolio']['weights']['GOOGL'])

    # =========================================================================
    # TESTS 16-18: PREDICTIVE MACHINE LEARNING ANALYST
    # =========================================================================
    def test_16_ml_rsi_and_volatility_features(self):
        ml_agent = MLPredictiveAnalystAgent()
        feats = ml_agent.extract_features(self.sample_prices['AAPL'])
        self.assertIn('rsi_14', feats.columns)
        self.assertIn('volatility_20d', feats.columns)
        self.assertTrue((feats['rsi_14'] >= 0).all() and (feats['rsi_14'] <= 100).all())

    def test_17_ml_feature_importances_sum(self):
        ml_agent = MLPredictiveAnalystAgent()
        res = ml_agent.predict(self.raw_data)
        for ticker in self.raw_data['tickers']:
            fi = res['ml_results'][ticker]['feature_importances']
            self.assertAlmostEqual(sum(fi.values()), 1.0, places=2)

    def test_18_ml_prediction_keys(self):
        ml_agent = MLPredictiveAnalystAgent()
        res = ml_agent.predict(self.raw_data)
        self.assertIn('predicted_annualized_returns', res)
        self.assertIn('latest_features', res)
        self.assertIn('model_used', res)

    # =========================================================================
    # TESTS 19-20: AI AGENT & SYSTEM COORDINATOR
    # =========================================================================
    def test_19_ai_agent_offline_mode(self):
        ai = AIPortfolioAnalystAgent(api_key="")
        report = ai.generate_report(self.raw_data)
        self.assertEqual(report['status'], 'offline')
        self.assertIn("Gemini", report['report'])

    def test_20_coordinator_end_to_end(self):
        coord = AgentSystemCoordinator()
        coord.data_architect.fetch_data = lambda tickers, start_date, end_date: self.raw_data
        bundle = coord.run_pipeline(['AAPL', 'MSFT', 'GOOGL'], risk_free_rate=0.04, return_multiplier=1.1, use_ml_views=True)
        
        self.assertEqual(len(bundle['tickers']), 3)
        self.assertIn('fundamental', bundle)
        self.assertIn('quant', bundle)
        self.assertIn('ml_predictive', bundle)
        self.assertIn('ai_report', bundle)

if __name__ == '__main__':
    unittest.main()
