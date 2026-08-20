import os
import json
import logging
from typing import Dict, Any

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logger = logging.getLogger("AIPortfolioAnalyst")

class AIPortfolioAnalystAgent:
    """
    Agent 6: Gemini AI Executive Summarizer
    Synthesizes outputs from Agent 1 (Data Architect), Agent 2 (Fundamental Analyst),
    Agent 3 (Quant Portfolio Optimizer), and Agent 4 (Predictive ML Analyst).
    Uses Google Gemini API to generate an Wall Street Investment Memo & Portfolio Rationale.
    """
    def __init__(self, api_key: str = None):
        self.name = "Gemini AI Executive Summarizer"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def generate_report(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize analytics bundle into a natural language executive report using Gemini AI.
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
                    "### 💡 Gemini AI Executive Report (Offline Mode)\n"
                    "To generate a real-time Wall Street AI Investment Memo synthesizing "
                    "Fundamental Ratios, Machine Learning Return Forecasts, and MPT Portfolio Weights, "
                    "please enter your free **Gemini API Key** in the sidebar!"
                )
            }

        try:
            genai.configure(api_key=self.api_key)
            # Try available Gemini models
            model_name = "gemini-2.5-flash"
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
You are a Lead Quant & Wall Street Investment Strategist.
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
