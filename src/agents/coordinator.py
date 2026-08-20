import logging
from typing import List, Dict, Any
import datetime

from src.agents.data_architect import DataArchitectAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.quant_analyst import QuantAnalystAgent
from src.agents.ml_predictive_analyst import MLPredictiveAnalystAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemCoordinator")

class AgentSystemCoordinator:
    """
    Master Orchestrator coordinating:
    - Agent 1: Data Architect
    - Agent 2: Fundamental Analyst
    - Agent 3: Quantitative Analyst
    - Agent 4: Predictive ML Analyst
    to pass compiled analytics to Agent 5 (Full-Stack Dashboard Developer).
    """
    def __init__(self):
        self.data_architect = DataArchitectAgent()
        self.fundamental_analyst = FundamentalAnalystAgent()
        self.quant_analyst = QuantAnalystAgent()
        self.ml_analyst = MLPredictiveAnalystAgent(model_type="random_forest")

    def run_pipeline(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
        risk_free_rate: float = 0.04,
        use_ml_forecast: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full multi-agent workflow sequentially:
        1. Data Architect -> Ingest clean data
        2. Fundamental Analyst -> Compute ratio metrics
        3. Predictive ML Analyst -> Forecast forward returns with Random Forest
        4. Quant Analyst -> Optimize portfolio with MPT
        """
        logger.info("Step 1: Financial Data Architect fetching & cleaning data...")
        raw_data = self.data_architect.fetch_data(tickers=tickers, start_date=start_date, end_date=end_date)

        logger.info("Step 2: Fundamental Analyst computing health metrics...")
        fundamental_output = self.fundamental_analyst.analyze(raw_data)

        logger.info("Step 3: Predictive ML Analyst training Random Forest regressor & forecasting returns...")
        ml_output = self.ml_analyst.predict(raw_data)

        logger.info("Step 4: Quantitative Analyst optimizing portfolio with MPT...")
        quant_output = self.quant_analyst.optimize_portfolio(raw_data, risk_free_rate=risk_free_rate)

        logger.info("Step 5: Compiling JSON analytics bundle for Full-Stack Dashboard Developer...")
        bundle = {
            'timestamp': datetime.datetime.now().isoformat(),
            'tickers': raw_data['tickers'],
            'start_date': raw_data['start_date'],
            'end_date': raw_data['end_date'],
            'prices': raw_data['prices'],
            'statements': raw_data['statements'],
            'fundamental': fundamental_output,
            'ml_predictive': ml_output,
            'quant': quant_output
        }

        return bundle
