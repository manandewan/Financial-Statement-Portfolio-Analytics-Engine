import pandas as pd
import yfinance as yf
from typing import List, Dict, Any
import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataArchitect")

class DataArchitectAgent:
    """
    Agent 1: Financial Data Architect
    Lead Data Engineer responsible for fetching, cleaning, and standardizing financial statement
    and price data for requested tickers over a given target date range with fault-tolerant retrieval.
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
            start_dt = datetime.date.today() - datetime.timedelta(days=5*365)
            start_date = start_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching price data for {tickers} from {start_date} to {end_date}")

        # Configure yfinance cache directory
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

        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame(name=tickers[0])

        # Clean missing price values
        adj_close = adj_close.ffill().bfill().dropna(how='all')

        # Statements for each ticker - Isolated robust fetching
        statements = {}
        for ticker in tickers:
            logger.info(f"Fetching statements for {ticker}")
            t_obj = yf.Ticker(ticker)
            
            # 1. Income Statement
            inc_stmt = pd.DataFrame()
            try:
                inc_stmt = t_obj.financials
                if inc_stmt is None or inc_stmt.empty:
                    inc_stmt = t_obj.quarterly_financials
            except Exception as e:
                logger.warning(f"Income statement error for {ticker}: {e}")

            # 2. Balance Sheet
            bal_sheet = pd.DataFrame()
            try:
                bal_sheet = t_obj.balance_sheet
                if bal_sheet is None or bal_sheet.empty:
                    bal_sheet = t_obj.quarterly_balance_sheet
            except Exception as e:
                logger.warning(f"Balance sheet error for {ticker}: {e}")

            # 3. Cash Flow
            cash_flow = pd.DataFrame()
            try:
                cash_flow = t_obj.cashflow
                if cash_flow is None or cash_flow.empty:
                    cash_flow = t_obj.quarterly_cashflow
            except Exception as e:
                logger.warning(f"Cash flow error for {ticker}: {e}")

            # 4. Fast Info & Info (with fallback)
            info = {}
            try:
                # Fast info is instant and doesn't rate-limit on Cloud IPs
                if hasattr(t_obj, 'fast_info'):
                    for k in ['market_cap', 'last_price', 'shares', 'year_high', 'year_low', 'currency']:
                        if hasattr(t_obj.fast_info, k):
                            info[k] = getattr(t_obj.fast_info, k)
                    if 'market_cap' in info:
                        info['marketCap'] = info['market_cap']
            except Exception:
                pass

            try:
                full_info = t_obj.info
                if isinstance(full_info, dict):
                    info.update(full_info)
            except Exception as e:
                logger.info(f"Full info lookup skipped for {ticker} (using fast_info & statements): {e}")

            statements[ticker] = {
                'income_statement': inc_stmt if inc_stmt is not None else pd.DataFrame(),
                'balance_sheet': bal_sheet if bal_sheet is not None else pd.DataFrame(),
                'cash_flow': cash_flow if cash_flow is not None else pd.DataFrame(),
                'info': info
            }

        return {
            'prices': adj_close,
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'statements': statements
        }
