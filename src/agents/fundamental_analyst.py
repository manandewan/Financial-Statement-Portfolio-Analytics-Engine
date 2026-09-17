import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger("FundamentalAnalyst")

class FundamentalAnalystAgent:
    """
    Agent 2: Fundamental Analyst
    Equity & Credit Researcher responsible for extracting:
    - Fundamental Corporate Health: Debt-to-Equity, Current Ratio, ROE, FCF Yield
    - Credit Worthiness & Solvency:
      * Leverage: Total Debt / EBITDA, Net Debt / EBITDA
      * Coverage: EBITDA / Interest Expense, Fixed Charge Coverage Ratio (FCCR)
      * Profitability & Cash Generation: Return on Equity (ROE), Free Cash Flow Yield
      * Credit Solvency & Debt Signals: Objective financial flags (Net Cash Surplus, Conservative Leverage, High Coverage, Cash Flow Repayment)
    Generates structured metrics and credit risk flags for each asset.
    """
    def __init__(self):
        self.name = "Fundamental Analyst"

    def _get_item(self, df: pd.DataFrame, possible_keys: List[str]) -> float:
        """
        Safely retrieve the most recent reported value for a set of possible line item keys.
        """
        if df is None or df.empty:
            return np.nan

        index_map = {str(idx).strip().lower(): idx for idx in df.index}
        
        for key in possible_keys:
            key_lower = key.strip().lower()
            matched_key = None
            if key_lower in index_map:
                matched_key = index_map[key_lower]
            else:
                for k_low, orig_key in index_map.items():
                    if key_lower in k_low:
                        matched_key = orig_key
                        break

            if matched_key is not None:
                series = df.loc[matched_key]
                if isinstance(series, pd.Series):
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
        Ingest raw data from Data Architect and compute financial health & credit worthiness metrics.
        """
        statements = raw_data.get('statements', {})
        tickers = raw_data.get('tickers', [])
        prices_df = raw_data.get('prices', pd.DataFrame())

        metrics_summary = {}
        credit_summary = {}

        sector_defaults = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Communication Services',
            'GOOG': 'Communication Services', 'AMZN': 'Consumer Cyclical', 'NVDA': 'Technology',
            'TSLA': 'Consumer Cyclical', 'META': 'Communication Services', 'NFLX': 'Communication Services',
            'JPM': 'Financial Services', 'JNJ': 'Healthcare', 'PG': 'Consumer Defensive',
            'WMT': 'Consumer Defensive', 'XOM': 'Energy', 'AMD': 'Technology', 'CRM': 'Technology',
            'RELIANCE.NS': 'Energy / Conglomerate', 'TCS.NS': 'Technology', 'INFY.NS': 'Technology',
            'HDFCBANK.NS': 'Financial Services', 'ICICIBANK.NS': 'Financial Services', 'TATAMOTORS.NS': 'Automotive',
            'RELIANCE.BO': 'Energy / Conglomerate', 'TCS.BO': 'Technology', 'INFY.BO': 'Technology'
        }

        for ticker in tickers:
            t_data = statements.get(ticker, {})
            inc = t_data.get('income_statement', pd.DataFrame())
            bal = t_data.get('balance_sheet', pd.DataFrame())
            cf = t_data.get('cash_flow', pd.DataFrame())
            info = t_data.get('info', {})

            # --- BALANCE SHEET ITEMS ---
            total_equity = self._get_item(bal, [
                'Total Stockholder Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest', 
                'Common Stock Equity', 'Total Equity', 'Total Stockholders Equity'
            ])
            total_debt = self._get_item(bal, [
                'Total Debt', 'Long Term Debt', 'Total Liab', 'Total Liabilities Net Minority Interest',
                'Long Term Debt And Capital Lease Obligation', 'Current Debt And Capital Lease Obligation'
            ])
            cash_and_equivalents = self._get_item(bal, [
                'Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments',
                'Cash Financial', 'Cash', 'Cash And Short Term Investments'
            ])
            total_assets = self._get_item(bal, [
                'Total Assets', 'Total Assets Net Minority Interest', 'Gross Assets'
            ])
            current_assets = self._get_item(bal, [
                'Current Assets', 'Total Current Assets'
            ])
            current_liab = self._get_item(bal, [
                'Current Liabilities', 'Total Current Liabilities'
            ])
            inventory = self._get_item(bal, [
                'Inventory', 'Total Inventory', 'Inventories'
            ])

            # --- INCOME STATEMENT ITEMS ---
            net_income = self._get_item(inc, [
                'Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operation',
                'Net Income From Continuing Operation Net Minority Interest'
            ])
            revenue = self._get_item(inc, [
                'Total Revenue', 'Operating Revenue', 'Gross Profit'
            ])
            operating_income = self._get_item(inc, [
                'Operating Income', 'Operating Revenue', 'EBIT'
            ])
            ebitda = self._get_item(inc, [
                'Normalized EBITDA', 'EBITDA', 'Ebitda'
            ])
            interest_expense = self._get_item(inc, [
                'Interest Expense', 'Interest Expense Non Operating', 'Total Interest Expense',
                'Net Non Operating Interest Income Expense'
            ])
            tax_provision = self._get_item(inc, [
                'Tax Provision', 'Income Tax Expense', 'Provision For Income Taxes'
            ])

            # --- CASH FLOW ITEMS ---
            free_cash_flow = self._get_item(cf, [
                'Free Cash Flow', 'Free Cashflow'
            ])
            operating_cash_flow = self._get_item(cf, [
                'Operating Cash Flow', 'Cash Flow From Operations', 'Operating Cashflow'
            ])
            capex = self._get_item(cf, [
                'Capital Expenditure', 'Capital Expenditures', 'Investing Cash Flow'
            ])
            depreciation = self._get_item(cf, [
                'Reconciled Depreciation', 'Depreciation & Amortization', 'Depreciation Amortization Depletion',
                'Depreciation And Amortization'
            ])
            taxes_paid = self._get_item(cf, [
                'Income Tax Paid Supplemental Data', 'Cash Taxes Paid', 'Taxes Paid'
            ])
            principal_repayments = self._get_item(cf, [
                'Repayment Of Debt', 'Long Term Debt Payments', 'Reduction Of Long Term Debt'
            ])
            lease_payments = self._get_item(cf, [
                'Finance Lease Payments', 'Operating Lease Payments', 'Payment For Lease'
            ])

            # --- FALLBACK CALCULATIONS ---
            capex_abs = abs(capex) if not pd.isna(capex) else 0.0

            # FCF fallback
            if pd.isna(free_cash_flow) and not pd.isna(operating_cash_flow):
                if not pd.isna(capex):
                    free_cash_flow = operating_cash_flow - capex_abs
                else:
                    free_cash_flow = operating_cash_flow

            # EBITDA fallback: Operating Income + Depreciation or from yfinance info
            if pd.isna(ebitda):
                if not pd.isna(operating_income) and not pd.isna(depreciation):
                    ebitda = operating_income + abs(depreciation)
                elif not pd.isna(operating_income):
                    ebitda = operating_income
                elif 'ebitda' in info and info['ebitda'] is not None:
                    ebitda = float(info['ebitda'])

            # Cash Taxes fallback
            cash_taxes = abs(taxes_paid) if not pd.isna(taxes_paid) else (abs(tax_provision) if not pd.isna(tax_provision) else 0.0)

            # Interest Expense fallback
            if pd.isna(interest_expense):
                if 'interestExpense' in info and info['interestExpense'] is not None:
                    interest_expense = abs(float(info['interestExpense']))
                else:
                    interest_expense = 0.0
            else:
                interest_expense = abs(interest_expense)

            # Cash fallback
            if pd.isna(cash_and_equivalents):
                if 'totalCash' in info and info['totalCash'] is not None:
                    cash_and_equivalents = float(info['totalCash'])
                else:
                    cash_and_equivalents = 0.0

            # Debt fallback
            if pd.isna(total_debt) and 'totalDebt' in info and info['totalDebt'] is not None:
                total_debt = float(info['totalDebt'])
            elif pd.isna(total_debt):
                total_debt = 0.0

            # Inventory fallback
            if pd.isna(inventory):
                inventory = 0.0

            # Fallbacks from yfinance info
            market_cap = info.get('marketCap', np.nan)
            latest_price = np.nan
            if ticker in prices_df.columns and not prices_df[ticker].dropna().empty:
                latest_price = float(prices_df[ticker].dropna().iloc[-1])

            if (pd.isna(market_cap) or market_cap == 0) and not pd.isna(latest_price):
                shares = info.get('sharesOutstanding') or info.get('shares') or info.get('impliedSharesOutstanding')
                if shares:
                    market_cap = latest_price * float(shares)

            if pd.isna(total_equity) and 'bookValue' in info and 'sharesOutstanding' in info:
                total_equity = info.get('bookValue', 0) * info.get('sharesOutstanding', 0)
            if pd.isna(current_assets) and 'totalCurrentAssets' in info:
                current_assets = info.get('totalCurrentAssets')
            if pd.isna(current_liab) and 'totalCurrentLiabilities' in info:
                current_liab = info.get('totalCurrentLiabilities')
            if pd.isna(net_income) and 'netIncomeToCommon' in info:
                net_income = info.get('netIncomeToCommon')
            if pd.isna(free_cash_flow) and 'freeCashflow' in info:
                free_cash_flow = info.get('freeCashflow')

            if pd.isna(total_assets):
                if 'totalAssets' in info and info['totalAssets'] is not None:
                    total_assets = float(info['totalAssets'])
                elif not pd.isna(total_equity) and not pd.isna(total_debt):
                    total_assets = float(total_debt + total_equity)
                elif not pd.isna(current_assets):
                    total_assets = float(current_assets)

            enterprise_value = info.get('enterpriseValue')
            if pd.isna(enterprise_value) or enterprise_value is None or enterprise_value <= 0:
                if not pd.isna(market_cap) and market_cap > 0:
                    c_val = cash_and_equivalents if not pd.isna(cash_and_equivalents) else 0.0
                    d_val = total_debt if not pd.isna(total_debt) else 0.0
                    enterprise_value = float(market_cap + d_val - c_val)
                else:
                    enterprise_value = np.nan
            else:
                enterprise_value = float(enterprise_value)

            # =========================================================
            # 1. CORE FUNDAMENTAL RATIOS (EQUITY)
            # =========================================================
            if not pd.isna(total_debt) and not pd.isna(total_equity) and total_equity != 0:
                debt_to_equity = float(total_debt / total_equity)
            elif 'debtToEquity' in info and info['debtToEquity'] is not None:
                debt_to_equity = float(info['debtToEquity']) / 100.0 if info['debtToEquity'] > 10 else float(info['debtToEquity'])
            else:
                debt_to_equity = np.nan

            if not pd.isna(current_assets) and not pd.isna(current_liab) and current_liab != 0:
                current_ratio = float(current_assets / current_liab)
            elif 'currentRatio' in info and info['currentRatio'] is not None:
                current_ratio = float(info['currentRatio'])
            else:
                current_ratio = np.nan

            if not pd.isna(net_income) and not pd.isna(total_equity) and total_equity != 0:
                roe = float(net_income / total_equity)
            elif 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                roe = float(info['returnOnEquity'])
            else:
                roe = np.nan

            if not pd.isna(free_cash_flow) and not pd.isna(market_cap) and market_cap > 0:
                fcf_yield = float(free_cash_flow / market_cap)
            else:
                fcf_yield = np.nan

            flags = []
            if not pd.isna(total_equity) and total_equity < 0:
                flags.append("Negative Equity Deficit")

            if not pd.isna(debt_to_equity):
                if debt_to_equity > 2.5:
                    flags.append("High Debt Leverage (D/E > 2.5)")
                elif 0 <= debt_to_equity < 1.0:
                    flags.append("Conservative Debt (D/E < 1.0)")

            if not pd.isna(current_ratio):
                if current_ratio < 1.0:
                    flags.append("Liquidity Strain (Current Ratio < 1.0)")
                elif current_ratio >= 1.5:
                    flags.append("Strong Liquidity Coverage (Current Ratio >= 1.5)")

            if not pd.isna(roe):
                if roe > 0.15:
                    flags.append("High Capital Efficiency (ROE > 15%)")
                elif roe < 0:
                    flags.append("Negative ROE (Unprofitable)")

            if not pd.isna(fcf_yield):
                if fcf_yield > 0.03:
                    flags.append("Strong FCF Yield (> 3%)")

            sector = info.get('sector')
            if not sector or sector == 'N/A':
                sector = sector_defaults.get(ticker, 'Equity Asset')

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
                'sector': sector,
                'industry': info.get('industry', 'N/A'),
                'flags': flags
            }

            # =========================================================
            # 2. CREDIT WORTHINESS & SOLVENCY RATIOS
            # =========================================================
            net_debt = float(total_debt - cash_and_equivalents) if not pd.isna(total_debt) and not pd.isna(cash_and_equivalents) else np.nan

            # A. LEVERAGE & CAPITAL STRUCTURE RATIOS
            if not pd.isna(total_debt) and not pd.isna(ebitda) and ebitda > 0:
                total_debt_to_ebitda = float(total_debt / ebitda)
            elif not pd.isna(total_debt) and total_debt == 0:
                total_debt_to_ebitda = 0.0
            else:
                total_debt_to_ebitda = np.nan

            if not pd.isna(net_debt) and not pd.isna(ebitda) and ebitda > 0:
                net_debt_to_ebitda = float(net_debt / ebitda)
            elif not pd.isna(net_debt) and net_debt <= 0:
                net_debt_to_ebitda = float(net_debt / ebitda) if not pd.isna(ebitda) and ebitda > 0 else 0.0
            else:
                net_debt_to_ebitda = np.nan

            # Market Loan-to-Value (LTV = Total Debt / Enterprise Value)
            if not pd.isna(total_debt) and not pd.isna(enterprise_value) and enterprise_value > 0:
                ltv = float(total_debt / enterprise_value)
            elif not pd.isna(total_debt) and not pd.isna(market_cap) and (market_cap + total_debt) > 0:
                ltv = float(total_debt / (market_cap + total_debt))
            elif not pd.isna(total_debt) and total_debt == 0:
                ltv = 0.0
            else:
                ltv = np.nan

            # Net LTV (Net Debt / Enterprise Value)
            if not pd.isna(net_debt) and not pd.isna(enterprise_value) and enterprise_value > 0:
                net_ltv = float(net_debt / enterprise_value)
            elif not pd.isna(net_debt) and net_debt <= 0:
                net_ltv = 0.0
            else:
                net_ltv = np.nan

            # Book LTV / Debt-to-Assets (Total Debt / Total Assets)
            if not pd.isna(total_debt) and not pd.isna(total_assets) and total_assets > 0:
                debt_to_assets = float(total_debt / total_assets)
            elif not pd.isna(total_debt) and total_debt == 0:
                debt_to_assets = 0.0
            else:
                debt_to_assets = np.nan

            # Debt-to-Capitalization (Total Debt / [Total Debt + Total Stockholder Equity])
            if not pd.isna(total_debt) and not pd.isna(total_equity) and (total_debt + total_equity) > 0:
                debt_to_capital = float(total_debt / (total_debt + total_equity))
            elif not pd.isna(total_debt) and total_debt == 0:
                debt_to_capital = 0.0
            else:
                debt_to_capital = np.nan

            # Financial Leverage Multiplier (Total Assets / Total Equity)
            if not pd.isna(total_assets) and not pd.isna(total_equity) and total_equity > 0:
                financial_leverage = float(total_assets / total_equity)
            else:
                financial_leverage = np.nan

            # B. COVERAGE RATIOS
            if not pd.isna(ebitda) and not pd.isna(interest_expense) and interest_expense > 0:
                ebitda_interest_coverage = float(ebitda / interest_expense)
            elif not pd.isna(interest_expense) and interest_expense == 0:
                ebitda_interest_coverage = 999.0  # Negligible interest burden
            else:
                ebitda_interest_coverage = np.nan

            # Fixed Charge Coverage Ratio (FCCR):
            fccr_numerator = (ebitda if not pd.isna(ebitda) else 0.0) - capex_abs - cash_taxes
            mand_repayments = abs(principal_repayments) if not pd.isna(principal_repayments) else 0.0
            lease_pmts = abs(lease_payments) if not pd.isna(lease_payments) else 0.0
            fccr_denominator = interest_expense + mand_repayments + lease_pmts

            if fccr_denominator > 0:
                fccr = float(fccr_numerator / fccr_denominator)
            elif interest_expense == 0 and total_debt == 0:
                fccr = 999.0  # Zero fixed charges
            else:
                fccr = np.nan

            # C. LIQUIDITY / SOLVENCY RATIOS
            if not pd.isna(operating_cash_flow) and not pd.isna(total_debt) and total_debt > 0:
                cfo_to_total_debt = float(operating_cash_flow / total_debt)
            elif not pd.isna(total_debt) and total_debt == 0:
                cfo_to_total_debt = 999.0
            else:
                cfo_to_total_debt = np.nan

            if not pd.isna(current_assets) and not pd.isna(current_liab) and current_liab > 0:
                quick_assets = current_assets - inventory
                quick_ratio = float(quick_assets / current_liab)
            elif 'quickRatio' in info and info['quickRatio'] is not None:
                quick_ratio = float(info['quickRatio'])
            else:
                quick_ratio = np.nan

            credit_flags = []
            if net_debt is not None and net_debt < 0:
                credit_flags.append("Net Cash Surplus")
            elif not pd.isna(net_debt_to_ebitda) and net_debt_to_ebitda > 3.5:
                credit_flags.append("High Leverage Warning (Net Debt/EBITDA > 3.5x)")
            elif not pd.isna(net_debt_to_ebitda) and net_debt_to_ebitda <= 1.5:
                credit_flags.append("Conservative Net Leverage (< 1.5x)")

            if not pd.isna(ltv):
                if 0 <= ltv < 0.15:
                    credit_flags.append("Ultra-Low LTV (< 15%)")
                elif 0.15 <= ltv < 0.30:
                    credit_flags.append("Conservative LTV (< 30%)")
                elif ltv > 0.60:
                    credit_flags.append("High LTV Exposure (> 60%)")

            if not pd.isna(debt_to_assets):
                if 0 <= debt_to_assets < 0.25:
                    credit_flags.append("Strong Asset Coverage (Debt/Assets < 25%)")
                elif debt_to_assets > 0.65:
                    credit_flags.append("High Asset Encumbrance (Debt/Assets > 65%)")

            if not pd.isna(ebitda_interest_coverage):
                if ebitda_interest_coverage >= 10.0:
                    credit_flags.append("Robust Interest Coverage (> 10x)")
                elif ebitda_interest_coverage < 2.0:
                    credit_flags.append("Low Interest Coverage (< 2.0x)")

            if not pd.isna(fccr):
                if fccr >= 3.0:
                    credit_flags.append("Strong Fixed Charge Buffer (FCCR >= 3.0x)")
                elif 0 <= fccr < 1.2:
                    credit_flags.append("Tight Debt Service Headroom (FCCR < 1.2x)")

            if not pd.isna(cfo_to_total_debt) and cfo_to_total_debt >= 0.30:
                credit_flags.append("Robust Cash Flow Repayment (CFO/Debt >= 30%)")

            credit_summary[ticker] = {
                'ebitda': ebitda,
                'total_debt': total_debt,
                'cash_and_equivalents': cash_and_equivalents,
                'net_debt': net_debt,
                'total_assets': total_assets,
                'enterprise_value': enterprise_value,
                'market_cap': market_cap,
                'interest_expense': interest_expense,
                'operating_cash_flow': operating_cash_flow,
                'capex': capex_abs,
                'cash_taxes': cash_taxes,
                'inventory': inventory,
                'total_debt_to_ebitda': total_debt_to_ebitda,
                'net_debt_to_ebitda': net_debt_to_ebitda,
                'ltv': ltv,
                'net_ltv': net_ltv,
                'debt_to_assets': debt_to_assets,
                'debt_to_capital': debt_to_capital,
                'financial_leverage': financial_leverage,
                'ebitda_interest_coverage': ebitda_interest_coverage,
                'fccr': fccr,
                'cfo_to_total_debt': cfo_to_total_debt,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                'credit_flags': credit_flags
            }

        return {
            'metrics': metrics_summary,
            'credit_metrics': credit_summary,
            'processed_count': len(metrics_summary)
        }
