import logging
from typing import List, Dict, Any
import datetime

from src.agents.data_architect import DataArchitectAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.quant_analyst import QuantAnalystAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemCoordinator")

class AgentSystemCoordinator:
    """
    Master Orchestrator coordinating Agent 1 (Data Architect), Agent 2 (Fundamental Analyst),
    and Agent 3 (Quantitative Analyst) to pass compiled analytics to Agent 4 (Full-Stack Dashboard Developer).
    """
    def __init__(self):
        self.data_architect = DataArchitectAgent()
        self.fundamental_analyst = FundamentalAnalystAgent()
        self.quant_analyst = QuantAnalystAgent()

    def run_pipeline(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
        risk_free_rate: float = 0.04
    ) -> Dict[str, Any]:
        """
        Execute full multi-agent workflow sequentially:
        1. Ingest clean data
        2. Analyze fundamentals
        3. Optimize portfolio
        """
        logger.info("Step 1: Financial Data Architect fetching & cleaning data...")
        raw_data = self.data_architect.fetch_data(tickers=tickers, start_date=start_date, end_date=end_date)

        logger.info("Step 2: Fundamental Analyst computing health metrics...")
        fundamental_output = self.fundamental_analyst.analyze(raw_data)

        logger.info("Step 3: Quantitative Analyst optimizing portfolio...")
        quant_output = self.quant_analyst.optimize_portfolio(raw_data, risk_free_rate=risk_free_rate)

        logger.info("Step 4: Compiling JSON analytics bundle for Full-Stack Dashboard Developer...")
        bundle = {
            'timestamp': datetime.datetime.now().isoformat(),
            'tickers': raw_data['tickers'],
            'start_date': raw_data['start_date'],
            'end_date': raw_data['end_date'],
            'prices': raw_data['prices'],
            'statements': raw_data['statements'],
            'fundamental': fundamental_output,
            'quant': quant_output
        }

        return bundle
