import pandas as pd
import yfinance as yf
from typing import List, Dict, Any
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataArchitect")

class DataArchitectAgent:
    """
    Agent 1: Financial Data Architect
    Lead Data Engineer responsible for fetching, cleaning, and standardizing financial statement
    and price data for requested tickers over a given target date range.
    """
    def __init__(self):
        self.name = "Financial Data Architect"

    def fetch_data(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Extract price history and financial statements for requested assets.
        """
        if not tickers:
            raise ValueError("Ticker list cannot be empty.")

        tickers = [t.strip().upper() for t in tickers if t.strip()]

        if end_date is None:
            end_date = datetime.date.today().strftime("%Y-%m-%d")
        if start_date is None:
            # Default to 5 years prior
            start_dt = datetime.date.today() - datetime.timedelta(days=5*365)
            start_date = start_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching price data for {tickers} from {start_date} to {end_date}")

        # Configure yfinance cache directory to avoid sqlite database locks
        import os
        cache_dir = os.path.expanduser("~/.cache/yf_custom_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            yf.set_tz_cache_location(cache_dir)
        except Exception:
            pass

        # Fetch historical prices with retry
        price_data = pd.DataFrame()
        for attempt in range(3):
            try:
                price_data = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)
                if not price_data.empty:
                    break
            except Exception as ex:
                logger.warning(f"Attempt {attempt+1} download failed: {ex}")

        if price_data.empty:
            # Fallback: fetch ticker by ticker
            df_list = {}
            for t in tickers:
                try:
                    t_df = yf.Ticker(t).history(start=start_date, end=end_date)
                    if not t_df.empty:
                        close_col = 'Adj Close' if 'Adj Close' in t_df.columns else 'Close'
                        df_list[t] = t_df[close_col]
                except Exception as ex:
                    logger.warning(f"Fallback download failed for {t}: {ex}")
            if df_list:
                price_data = pd.DataFrame(df_list)

        # Handle single vs multiple tickers from yfinance
        if isinstance(price_data.columns, pd.MultiIndex):
            if 'Adj Close' in price_data.columns.levels[0]:
                adj_close = price_data['Adj Close']
            elif 'Close' in price_data.columns.levels[0]:
                adj_close = price_data['Close']
            else:
                adj_close = price_data.iloc[:, :len(tickers)]
        else:
            if 'Adj Close' in price_data.columns:
                adj_close = price_data[['Adj Close']]
                adj_close.columns = tickers
            elif 'Close' in price_data.columns:
                adj_close = price_data[['Close']]
                adj_close.columns = tickers
            else:
                adj_close = price_data

        # Ensure DataFrame layout
        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame(name=tickers[0])

        # Clean missing price values
        adj_close = adj_close.ffill().bfill().dropna(how='all')

        # Statements for each ticker
        statements = {}
        for ticker in tickers:
            logger.info(f"Fetching statements for {ticker}")
            t_obj = yf.Ticker(ticker)
            
            try:
                inc_stmt = t_obj.financials
                bal_sheet = t_obj.balance_sheet
                cash_flow = t_obj.cashflow
                info = t_obj.info if hasattr(t_obj, 'info') else {}
            except Exception as e:
                logger.warning(f"Error fetching statements for {ticker}: {e}")
                inc_stmt = pd.DataFrame()
                bal_sheet = pd.DataFrame()
                cash_flow = pd.DataFrame()
                info = {}

            statements[ticker] = {
                'income_statement': inc_stmt,
                'balance_sheet': bal_sheet,
                'cash_flow': cash_flow,
                'info': info
            }

        return {
            'prices': adj_close,
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'statements': statements
        }
