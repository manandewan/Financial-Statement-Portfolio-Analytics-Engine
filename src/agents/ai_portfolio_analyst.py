import os
import json
import logging
import warnings
from typing import Dict, Any

# Suppress library deprecation warnings for clean demo presentations
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

HAS_GENAI = False
USE_NEW_SDK = False

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
    USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        HAS_GENAI = True
        USE_NEW_SDK = False
    except ImportError:
        HAS_GENAI = False

logger = logging.getLogger("AIPortfolioAnalyst")

class AIPortfolioAnalystAgent:
    """
    Agent 5: Gemini AI Executive Summarizer
    Synthesizes outputs from Agent 1 (Data Architect), Agent 2 (Fundamental Analyst),
    Agent 3 (Quant Portfolio Optimizer), and Agent 4 (Predictive ML Analyst).
    Uses Google Gemini 3.7 Flash API to generate an Executive Wall Street Investment Memo.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.7-flash"):
        self.name = "Gemini AI Executive Summarizer"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name

    def generate_report(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize analytics bundle into a natural language executive report using Gemini 3.7.
        """
        tickers = bundle.get('tickers', [])
        fund_metrics = bundle.get('fundamental', {}).get('metrics', {})
        ml_results = bundle.get('ml_predictive', {}).get('ml_results', {})
        quant = bundle.get('quant', {})
        max_sharpe = quant.get('max_sharpe_portfolio', {})
        min_var = quant.get('min_variance_portfolio', {})

        if not self.api_key or not HAS_GENAI:
            return {
                'status': 'offline',
                'report': (
                    "### 💡 Gemini 3.7 AI Executive Report (Offline Mode)\n\n"
                    "To generate an automated, real-time Wall Street Investment Memo synthesizing "
                    "**Fundamental Health Ratios**, **Machine Learning Forward Return Forecasts**, and **MPT Portfolio Weights**, "
                    "enter your **Gemini API Key** in the sidebar!"
                )
            }

        prompt = f"""
You are a Lead Quant & Wall Street Investment Strategist powered by Gemini 3.7.
Analyze the following multi-agent financial analytics data bundle for stocks: {tickers}.

DATA SUMMARY:
1. Fundamental Ratios (Agent 2):
{json.dumps(fund_metrics, indent=2, default=str)}

2. Predictive ML 20-Day Return Forecasts & Technical Indicators (Agent 4):
{json.dumps(ml_results, indent=2, default=str)}

3. Quantitative MPT Portfolio Optimization (Agent 3):
- Max Sharpe Portfolio Weights: {json.dumps(max_sharpe.get('weights', {}), indent=2)}
- Max Sharpe Expected Return: {max_sharpe.get('expected_return', 0)*100:.2f}%
- Max Sharpe Volatility: {max_sharpe.get('volatility', 0)*100:.2f}%
- Max Sharpe Ratio: {max_sharpe.get('sharpe_ratio', 0):.2f}
- Min Variance Portfolio Weights: {json.dumps(min_var.get('weights', {}), indent=2)}

TASK:
Write an **Executive Portfolio Investment Memo** covering:
1. **Executive Summary & Key Takeaways**: Highlight the top-performing assets and corporate health signals.
2. **Fundamental Health & Risk Analysis**: Note any red flags (High Debt D/E, Liquidity strains) or strong ROE/FCF champions.
3. **Predictive ML & Technical Signal Evaluation**: Explain the Random Forest return forecasts and RSI/momentum trends.
4. **Strategic Portfolio Allocation Rationale**: Explain WHY the Modern Portfolio Theory (MPT) optimization selected specific weights for the Max Sharpe portfolio.
5. **Actionable Rebalancing Recommendation**: Provide clear recommendations for an investor.

Format the response cleanly in GitHub Markdown with emojis and bold section headings. Keep it concise, analytical, and professional.
"""

        try:
            if USE_NEW_SDK:
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                report_text = response.text
            else:
                genai_legacy.configure(api_key=self.api_key)
                model = None
                models_to_try = [self.model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                for m_name in models_to_try:
                    try:
                        model = genai_legacy.GenerativeModel(m_name)
                        break
                    except Exception:
                        continue
                if model is None:
                    model = genai_legacy.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                report_text = response.text if hasattr(response, 'text') else str(response)

            return {
                'status': 'success',
                'report': report_text
            }

        except Exception as e:
            logger.warning(f"Error calling Gemini API: {e}")
            return {
                'status': 'error',
                'report': f"⚠️ **Gemini AI API Note**: {str(e)}\n\nPlease verify your Gemini API key in the sidebar."
            }
