import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger("FundamentalAnalyst")

class FundamentalAnalystAgent:
    """
    Agent 2: Fundamental Analyst
    Equity Researcher responsible for extracting corporate health metrics:
    - Debt-to-Equity Ratio
    - Current Ratio
    - Return on Equity (ROE)
    - Free Cash Flow (FCF) Yield
    Generates structured financial health metrics and warning flags for each asset.
    """
    def __init__(self):
        self.name = "Fundamental Analyst"

    def _get_item(self, df: pd.DataFrame, possible_keys: List[str]) -> float:
        """
        Safely retrieve the most recent reported value for a set of possible line item keys.
        """
        if df is None or df.empty:
            return np.nan

        # Lowercase index mapping
        index_map = {str(idx).strip().lower(): idx for idx in df.index}
        
        for key in possible_keys:
            key_lower = key.strip().lower()
            # Exact or partial match
            matched_key = None
            if key_lower in index_map:
                matched_key = index_map[key_lower]
            else:
                # substring search
                for k_low, orig_key in index_map.items():
                    if key_lower in k_low:
                        matched_key = orig_key
                        break

            if matched_key is not None:
                series = df.loc[matched_key]
                if isinstance(series, pd.Series):
                    # Pick most recent non-NaN value
                    valid_vals = series.dropna()
                    if len(valid_vals) > 0:
                        val = valid_vals.iloc[0]
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            continue
                elif isinstance(series, (int, float, np.number)):
                    return float(series)

        return np.nan

    def analyze(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest raw data from Data Architect and compute financial health metrics.
        """
        statements = raw_data.get('statements', {})
        tickers = raw_data.get('tickers', [])
        prices_df = raw_data.get('prices', pd.DataFrame())

        metrics_summary = {}

        for ticker in tickers:
            t_data = statements.get(ticker, {})
            inc = t_data.get('income_statement', pd.DataFrame())
            bal = t_data.get('balance_sheet', pd.DataFrame())
            cf = t_data.get('cash_flow', pd.DataFrame())
            info = t_data.get('info', {})

            # Extract balance sheet metrics
            total_equity = self._get_item(bal, [
                'Total Stockholder Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity'
            ])
            total_debt = self._get_item(bal, [
                'Total Debt', 'Long Term Debt', 'Total Liab', 'Total Liabilities Net Minority Interest'
            ])
            current_assets = self._get_item(bal, [
                'Current Assets', 'Total Current Assets'
            ])
            current_liab = self._get_item(bal, [
                'Current Liabilities', 'Total Current Liabilities'
            ])

            # Extract income statement metrics
            net_income = self._get_item(inc, [
                'Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operation'
            ])
            revenue = self._get_item(inc, [
                'Total Revenue', 'Operating Revenue'
            ])

            # Extract cash flow metrics
            free_cash_flow = self._get_item(cf, [
                'Free Cash Flow', 'Operating Cash Flow'
            ])
            operating_cash_flow = self._get_item(cf, [
                'Operating Cash Flow', 'Cash Flow From Operations'
            ])

            # Fallbacks from yfinance info if statement items missing
            market_cap = info.get('marketCap', np.nan)
            latest_price = np.nan
            if ticker in prices_df.columns and not prices_df[ticker].dropna().empty:
                latest_price = float(prices_df[ticker].dropna().iloc[-1])

            if pd.isna(market_cap) and not pd.isna(latest_price) and 'sharesOutstanding' in info:
                market_cap = latest_price * info.get('sharesOutstanding', 0)

            # Fallbacks from info dict if balance sheet parsing yielded NaN
            if pd.isna(total_equity) and 'bookValue' in info and 'sharesOutstanding' in info:
                total_equity = info.get('bookValue', 0) * info.get('sharesOutstanding', 0)
            if pd.isna(total_debt) and 'totalDebt' in info:
                total_debt = info.get('totalDebt')
            if pd.isna(current_assets) and 'totalCurrentAssets' in info:
                current_assets = info.get('totalCurrentAssets')
            if pd.isna(current_liab) and 'totalCurrentLiabilities' in info:
                current_liab = info.get('totalCurrentLiabilities')
            if pd.isna(net_income) and 'netIncomeToCommon' in info:
                net_income = info.get('netIncomeToCommon')
            if pd.isna(free_cash_flow) and 'freeCashflow' in info:
                free_cash_flow = info.get('freeCashflow')

            # Calculate Ratios
            # 1. Debt-to-Equity Ratio
            if not pd.isna(total_debt) and not pd.isna(total_equity) and total_equity != 0:
                debt_to_equity = float(total_debt / total_equity)
            elif 'debtToEquity' in info and info['debtToEquity'] is not None:
                debt_to_equity = float(info['debtToEquity']) / 100.0 if info['debtToEquity'] > 10 else float(info['debtToEquity'])
            else:
                debt_to_equity = np.nan

            # 2. Current Ratio
            if not pd.isna(current_assets) and not pd.isna(current_liab) and current_liab != 0:
                current_ratio = float(current_assets / current_liab)
            elif 'currentRatio' in info and info['currentRatio'] is not None:
                current_ratio = float(info['currentRatio'])
            else:
                current_ratio = np.nan

            # 3. Return on Equity (ROE)
            if not pd.isna(net_income) and not pd.isna(total_equity) and total_equity != 0:
                roe = float(net_income / total_equity)
            elif 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                roe = float(info['returnOnEquity'])
            else:
                roe = np.nan

            # 4. Free Cash Flow Yield
            if not pd.isna(free_cash_flow) and not pd.isna(market_cap) and market_cap > 0:
                fcf_yield = float(free_cash_flow / market_cap)
            else:
                fcf_yield = np.nan

            # Assessment / Flags
            flags = []
            if not pd.isna(debt_to_equity):
                if debt_to_equity > 2.5:
                    flags.append("High Debt Leverage (D/E > 2.5)")
                elif debt_to_equity < 1.0:
                    flags.append("Conservative Debt Structure (D/E < 1.0)")

            if not pd.isna(current_ratio):
                if current_ratio < 1.0:
                    flags.append("Liquidity Strain (Current Ratio < 1.0)")
                elif current_ratio >= 1.5:
                    flags.append("Strong Liquidity Coverage (Current Ratio >= 1.5)")

            if not pd.isna(roe):
                if roe > 0.15:
                    flags.append("High Return on Equity (ROE > 15%)")
                elif roe < 0:
                    flags.append("Negative ROE (Unprofitable)")

            if not pd.isna(fcf_yield):
                if fcf_yield > 0.05:
                    flags.append("Strong FCF Yield (> 5%)")

            metrics_summary[ticker] = {
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'return_on_equity': roe,
                'free_cash_flow_yield': fcf_yield,
                'market_cap': market_cap,
                'net_income': net_income,
                'total_equity': total_equity,
                'total_debt': total_debt,
                'free_cash_flow': free_cash_flow,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'flags': flags
            }

        return {
            'metrics': metrics_summary,
            'processed_count': len(metrics_summary)
        }
