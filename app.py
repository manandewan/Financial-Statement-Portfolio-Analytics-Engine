import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json
import os
import html

from src.agents.coordinator import AgentSystemCoordinator

# Page Configuration - Responsive & Mobile-Ready with Custom Logo
st.set_page_config(
    page_title="Financial Statement & Portfolio Analytics Engine",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global Plotly Mobile-Lock Configuration (Disables touch-zoom hijacking & toolbar overlap)
PLOTLY_CONFIG = {
    'displayModeBar': False,          # Hides floating toolbar completely
    'scrollZoom': False,              # Disables scroll/pinch zooming
    'showAxisDragHandles': False,     # Disables dragging on axes
    'showAxisRangeEntryBoxes': False, # Disables range entry boxes
    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
}

def lock_chart_for_mobile(fig):
    """
    Locks axes and drag modes so touching on mobile scrolls the page naturally
    without distorting chart axes or hijacking finger gestures.
    """
    fig.update_layout(
        dragmode=False,
        margin=dict(l=10, r=10, t=40, b=20),
        hovermode="closest"
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

# Custom Styling (Mobile-Responsive & Modern Dark Aesthetic)
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .sub-header {
        color: #A0AEC0;
        font-size: 1rem;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }
    .agent-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.4rem;
    }
    .pill-data { background-color: #2B6CB0; color: white; }
    .pill-fund { background-color: #2F855A; color: white; }
    .pill-credit { background-color: #C53030; color: white; }
    .pill-quant { background-color: #6B46C1; color: white; }
    .pill-ml { background-color: #D69E2E; color: black; }
    .pill-ai { background-color: #E53E3E; color: white; }
    .pill-dev { background-color: #DD6B20; color: white; }

    .credit-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-left: 5px solid #38BDF8;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .responsive-table-wrapper {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px solid #334155;
        border-radius: 8px;
        background: #0F172A;
        margin-top: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .styled-finance-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .styled-finance-table th {
        background: #1E293B;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        padding: 10px 12px;
        border-bottom: 2px solid #334155;
        text-align: right;
    }
    .styled-finance-table th.ticker-col, .styled-finance-table th.wrap-col {
        text-align: left;
    }
    .styled-finance-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #1E293B;
        vertical-align: middle;
    }
    .styled-finance-table td.ticker-col {
        font-weight: 700;
        color: #38BDF8;
        white-space: nowrap;
        text-align: left;
    }
    .styled-finance-table td.metric-col {
        white-space: nowrap;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .styled-finance-table td.wrap-col {
        white-space: normal !important;
        word-wrap: break-word !important;
        min-width: 280px;
        max-width: 480px;
        text-align: left;
        line-height: 1.45;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        align-items: center;
    }
    .signal-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 500;
        line-height: 1.3;
        white-space: normal;
    }
    .badge-success {
        background-color: rgba(34, 197, 94, 0.16);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.16);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .styled-finance-table th.benchmark-col-green, .styled-finance-table td.benchmark-col-green {
        color: #4ADE80;
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
    }
    .styled-finance-table th.benchmark-col-amber, .styled-finance-table td.benchmark-col-amber {
        color: #FBBF24;
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
    }
    .styled-finance-table th.benchmark-col-red, .styled-finance-table td.benchmark-col-red {
        color: #F87171;
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
    }
    .styled-finance-table td.formula-col {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.78rem;
        color: #93C5FD;
        white-space: nowrap;
    }

    @media (max-width: 768px) {
        .main-header {
            font-size: 1.4rem !important;
        }
        .sub-header {
            font-size: 0.85rem !important;
        }
        .agent-pill {
            font-size: 0.65rem !important;
            padding: 0.2rem 0.45rem !important;
            margin-bottom: 0.3rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def render_styled_table(df: pd.DataFrame, wrap_cols=None, badge_cols=None) -> str:
    """
    Renders a responsive, modern HTML table that properly wraps long text
    and displays signals/alerts as styled badges rather than truncating them with ellipses.
    """
    if wrap_cols is None:
        wrap_cols = []
    if badge_cols is None:
        badge_cols = []

    html_out = ['<div class="responsive-table-wrapper"><table class="styled-finance-table">']
    html_out.append('<thead><tr>')
    for col in df.columns:
        col_str = str(col)
        if col_str in wrap_cols:
            cls = 'wrap-col'
        elif col_str in ['Ticker', 'Credit Ratio']:
            cls = 'ticker-col'
        elif any(sym in col_str for sym in ['🟢', 'Prime']):
            cls = 'benchmark-col-green'
        elif any(sym in col_str for sym in ['🟡', 'Moderate']):
            cls = 'benchmark-col-amber'
        elif any(sym in col_str for sym in ['🔴', 'Caution', 'High Risk']):
            cls = 'benchmark-col-red'
        else:
            cls = 'metric-col'
        html_out.append(f'<th class="{cls}">{html.escape(col_str)}</th>')
    html_out.append('</tr></thead><tbody>')

    for _, row in df.iterrows():
        html_out.append('<tr>')
        for col in df.columns:
            col_str = str(col)
            val = str(row[col])
            if col_str in badge_cols:
                badges = []
                parts = [p.strip() for p in val.split(',') if p.strip()]
                for p in parts:
                    p_lower = p.lower()
                    if any(w in p_lower for w in ['tight', 'high leverage', 'low interest', 'negative', 'strain', 'warning', 'deficit', 'unprofitable', 'elevated', 'encumbrance']):
                        b_cls = 'signal-badge badge-warning'
                    elif any(s in p_lower for s in ['conservative', 'robust', 'strong', 'prime', 'net cash', 'efficiency', 'high capital', 'ultra-low']):
                        b_cls = 'signal-badge badge-success'
                    else:
                        b_cls = 'signal-badge badge-neutral'
                    badges.append(f'<span class="{b_cls}">{html.escape(p)}</span>')
                content = '<div class="badge-container">' + ' '.join(badges) + '</div>' if badges else html.escape(val)
                html_out.append(f'<td class="wrap-col">{content}</td>')
            elif col_str in wrap_cols:
                html_out.append(f'<td class="wrap-col">{html.escape(val)}</td>')
            elif col_str in ['Ticker', 'Credit Ratio']:
                html_out.append(f'<td class="ticker-col">{html.escape(val)}</td>')
            elif any(sym in col_str for sym in ['🟢', 'Prime']):
                html_out.append(f'<td class="benchmark-col-green">{html.escape(val)}</td>')
            elif any(sym in col_str for sym in ['🟡', 'Moderate']):
                html_out.append(f'<td class="benchmark-col-amber">{html.escape(val)}</td>')
            elif any(sym in col_str for sym in ['🔴', 'Caution', 'High Risk']):
                html_out.append(f'<td class="benchmark-col-red">{html.escape(val)}</td>')
            elif col_str == 'Formula':
                html_out.append(f'<td class="formula-col"><code>{html.escape(val)}</code></td>')
            else:
                html_out.append(f'<td class="metric-col">{html.escape(val)}</td>')
        html_out.append('</tr>')
    html_out.append('</tbody></table></div>')
    return ''.join(html_out)


@st.cache_data(ttl=3600, show_spinner=False)
def run_agent_pipeline(tickers_tuple, start_str, end_str, rf_rate, ret_multiplier, use_ml_views, gemini_key, shrink_returns=False, max_asset_weight=None):
    """
    Cached helper to execute multi-agent coordinator pipeline.
    """
    coordinator = AgentSystemCoordinator(gemini_api_key=gemini_key)
    return coordinator.run_pipeline(
        tickers=list(tickers_tuple),
        start_date=start_str,
        end_date=end_str,
        risk_free_rate=rf_rate,
        return_multiplier=ret_multiplier,
        use_ml_views=use_ml_views,
        gemini_api_key=gemini_key,
        shrink_returns=shrink_returns,
        max_asset_weight=max_asset_weight
    )

def main():
    # Sidebar Configuration
    if os.path.exists("assets/logo.png"):
        st.sidebar.image("assets/logo.png", width=70)
    st.sidebar.title("⚙️ Dashboard Controls")

    preset_options = {
        "Big Tech Leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "Diversified Blue Chips": ["JPM", "JNJ", "PG", "WMT", "XOM"],
        "Growth & Tech": ["TSLA", "AMD", "META", "NFLX", "CRM"],
        "Custom Input": []
    }

    selected_preset = st.sidebar.selectbox("Select Portfolio Preset", list(preset_options.keys()))

    if selected_preset != "Custom Input":
        default_tickers = preset_options[selected_preset]
        ticker_input = st.sidebar.text_input("Stock Tickers (comma-separated)", ", ".join(default_tickers))
    else:
        ticker_input = st.sidebar.text_input("Stock Tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, NVDA")

    ticker_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    # Date range selection
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        default_start = datetime.date.today() - datetime.timedelta(days=5*365)
        start_date = st.date_input("Start Date", default_start)
    with col_s2:
        end_date = st.date_input("End Date", datetime.date.today())

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quantitative & ML Settings")
    
    # Risk-free rate
    rf_rate_pct = st.sidebar.slider("Risk-Free Rate ($R_f$ %)", min_value=0.0, max_value=12.0, value=4.0, step=0.25)
    rf_rate = rf_rate_pct / 100.0

    # Expected Return Adjustment Slider
    return_shift_pct = st.sidebar.slider("Expected Return Adjustment (%)", min_value=-50, max_value=100, value=0, step=5,
                                        help="Scale expected future asset returns. Affects all MPT allocations, Efficient Frontier curves, and Sharpe ratios.")
    ret_multiplier = 1.0 + (return_shift_pct / 100.0)

    # ML-Enhanced Return Views Toggle
    use_ml_views = st.sidebar.checkbox("🤖 Use Random Forest ML Views in MPT", value=False,
                                       help="Replaces static historical returns with Supervised ML forward return forecasts inside the Modern Portfolio Theory (MPT) optimizer.")

    # James-Stein Return Regularization Toggle
    shrink_returns = st.sidebar.checkbox("🛡️ Apply James-Stein Return Regularization", value=True,
                                        help="Applies Bayes-Stein shrinkage to pull noisy, extreme historical sample returns toward the 10% market equilibrium prior, dampening Michaud error-maximization.")

    # Maximum Asset Allocation Limit (Concentration Cap)
    max_weight_pct = st.sidebar.slider(
        "Max Asset Position Limit (%)", 
        min_value=20, 
        max_value=100, 
        value=35, 
        step=5,
        help="Institutional concentration limit per asset (default: 35%). Prevents single-stock 100% corner solutions and enforces true multi-asset risk diversification to eliminate massive portfolio variance."
    )
    max_asset_weight = max_weight_pct / 100.0 if max_weight_pct < 100 else None

    # Optional Gemini API Key
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 Gemini AI Integration (Optional)")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Multi-Agent Architecture")
    st.sidebar.markdown("""
    - **Agent 1: Data Architect**: Ingests prices & statements.
    - **Agent 2: Fundamental & Credit Analyst**: Computes D/E, ROE, FCCR, Leverage & Solvency.
    - **Agent 3: Quantitative Analyst**: MPT, VaR/CVaR & Efficient Frontier.
    - **Agent 4: Predictive ML Analyst**: Random Forest return forecasts.
    - **Agent 5: Gemini AI Summarizer**: Wall Street Investment Memos.
    - **Agent 6: Dashboard Developer**: Streamlit & Plotly UI.
    """)

    st.sidebar.button("🚀 Run Agent Pipeline", type="primary", use_container_width=True)

    # App Header with Logo
    col_h1, col_h2 = st.columns([1, 14])
    with col_h1:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=65)
    with col_h2:
        st.markdown('<div class="main-header">Financial Statement & Portfolio Analytics Engine</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Multi-Agent Machine Learning, Credit Worthiness (FCCR/Leverage), Risk Analytics & MPT Platform</div>', unsafe_allow_html=True)

    # Pipeline Agent Badge Indicator
    st.markdown("""
    <div class="agent-indicator-bar">
        <span class="agent-pill">Data Ingestion: Complete</span>
        <span class="agent-pill">Fundamental Analytics: Complete</span>
        <span class="agent-pill">Credit Worthiness & Solvency: Complete</span>
        <span class="agent-pill">Predictive ML: Complete</span>
        <span class="agent-pill">Quant MPT & Tail Risk: Complete</span>
        <span class="agent-pill">Executive Synthesis: Ready</span>
    </div>
    """, unsafe_allow_html=True)

    # Validation Checks
    if not ticker_list:
        st.error("Please provide at least one valid stock ticker symbol.")
        st.stop()

    if start_date >= end_date:
        st.error("Start Date must precede End Date.")
        st.stop()

    # Execution Trigger
    with st.spinner("🔄 Coordinating Agents: Ingesting data, analyzing fundamentals & credit worthiness, training ML models, and optimizing portfolio..."):
        try:
            data_bundle = run_agent_pipeline(
                tickers_tuple=tuple(ticker_list),
                start_str=start_date.strftime("%Y-%m-%d"),
                end_str=end_date.strftime("%Y-%m-%d"),
                rf_rate=rf_rate,
                ret_multiplier=ret_multiplier,
                use_ml_views=use_ml_views,
                gemini_key=gemini_key,
                shrink_returns=shrink_returns,
                max_asset_weight=max_asset_weight
            )
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")
            st.stop()

    # Unpack Bundle
    tickers = data_bundle['tickers']
    prices_df = data_bundle['prices']
    statements = data_bundle['statements']
    fundamental_res = data_bundle['fundamental']['metrics']
    credit_res = data_bundle['fundamental'].get('credit_metrics', {})
    ml_res = data_bundle.get('ml_predictive', {})
    quant_res = data_bundle['quant']
    ai_report_res = data_bundle.get('ai_report', {})

    # Tabs Navigation (Including New Credit Worthiness & Solvency Tab)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏢 Fundamental Health",
        "💳 Credit Worthiness & Solvency",
        "📊 Historical Performance & Risk",
        "🤖 ML Return Forecasting",
        "🎯 Portfolio Optimization & VaR",
        "📝 Gemini AI Report"
    ])

    # ----------------------------------------------------
    # TAB 1: FUNDAMENTAL HEALTH
    # ----------------------------------------------------
    with tab1:
        st.subheader("Fundamental Corporate Health Metrics")
        st.markdown("Extracted by **Agent 2 (Fundamental Analyst)** from Balance Sheets, Income Statements, and Cash Flow Statements.")

        # Summary Table of Fundamental Metrics
        fund_table_data = []
        for t in tickers:
            m = fundamental_res.get(t, {})
            de = f"{m.get('debt_to_equity', np.nan):.2f}" if not pd.isna(m.get('debt_to_equity')) else "N/A"
            cr = f"{m.get('current_ratio', np.nan):.2f}" if not pd.isna(m.get('current_ratio')) else "N/A"
            roe = f"{m.get('return_on_equity', np.nan)*100:.2f}%" if not pd.isna(m.get('return_on_equity')) else "N/A"
            fcf_y = f"{m.get('free_cash_flow_yield', np.nan)*100:.2f}%" if not pd.isna(m.get('free_cash_flow_yield')) else "N/A"
            mcap = f"${m.get('market_cap', 0)/1e9:.2f}B" if m.get('market_cap') and not pd.isna(m.get('market_cap')) else "N/A"
            
            fund_table_data.append({
                "Ticker": t,
                "Sector": m.get('sector', 'N/A'),
                "Market Cap": mcap,
                "Debt-to-Equity [<1.0x]": de,
                "Current Ratio [>1.5x]": cr,
                "Return on Equity (ROE) [>15%]": roe,
                "FCF Yield [>3.0%]": fcf_y,
                "Health Alerts": ", ".join(m.get('flags', [])) if m.get('flags') else "Normal"
            })

        fund_df = pd.DataFrame(fund_table_data)
        st.markdown(render_styled_table(fund_df, wrap_cols=["Health Alerts", "Sector"], badge_cols=["Health Alerts"]), unsafe_allow_html=True)

        # Comparative Metrics Bar Charts
        st.markdown("### Comparative Ratio Charts")
        c1, c2 = st.columns(2)

        with c1:
            roe_vals = [fundamental_res.get(t, {}).get('return_on_equity', 0) or 0 for t in tickers]
            fig_roe = px.bar(
                x=tickers, y=[v*100 for v in roe_vals],
                labels={'x': 'Ticker', 'y': 'ROE (%)'},
                title="Return on Equity (ROE %)",
                color=[v*100 for v in roe_vals],
                color_continuous_scale="Viridis"
            )
            fig_roe = lock_chart_for_mobile(fig_roe)
            st.plotly_chart(fig_roe, use_container_width=True, config=PLOTLY_CONFIG)

        with c2:
            de_vals = [fundamental_res.get(t, {}).get('debt_to_equity', 0) or 0 for t in tickers]
            fig_de = px.bar(
                x=tickers, y=de_vals,
                labels={'x': 'Ticker', 'y': 'D/E Ratio'},
                title="Debt-to-Equity Ratio",
                color=de_vals,
                color_continuous_scale="Reds"
            )
            fig_de = lock_chart_for_mobile(fig_de)
            st.plotly_chart(fig_de, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### 🎯 Fundamental Equity Health Benchmarks")
        f_b1, f_b2, f_b3 = st.columns(3)
        with f_b1:
            st.success("**🟢 Fortress Equity Health**\n- Debt-to-Equity: $< 1.0x$\n- Current Ratio: $\\ge 1.5x$\n- ROE: $> 15.0\\%$\n- FCF Yield: $> 3.5\\%$\n- *High capital return efficiency, self-funding growth, low bankruptcy risk.*")
        with f_b2:
            st.warning("**🟡 Moderate / Stable Range**\n- Debt-to-Equity: $1.0x - 2.5x$\n- Current Ratio: $1.0x - 1.5x$\n- ROE: $8.0\\% - 15.0\\%$\n- FCF Yield: $1.5\\% - 3.5\\%$\n- *Acceptable financial flexibility; typical for mature industrial or utility firms.*")
        with f_b3:
            st.error("**🔴 High Risk / Distress Signs**\n- Debt-to-Equity: $> 2.5x$ or Deficit\n- Current Ratio: $< 1.0x$\n- ROE: $< 0.0\\%$ (Net Losses)\n- FCF Yield: $< 0.0\\%$ (Cash Burn)\n- *Working capital deficit, sustained shareholder dilution, or debt overhang.*")

        st.markdown("---")
        st.subheader("Financial Statement Deep Dive")
        selected_stmt_ticker = st.selectbox("Select Ticker to Inspect Raw Statements", tickers)
        
        if selected_stmt_ticker in statements:
            t_stmt = statements[selected_stmt_ticker]
            stmt_choice = st.radio("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
            
            if stmt_choice == "Income Statement":
                df_display = t_stmt.get('income_statement', pd.DataFrame())
            elif stmt_choice == "Balance Sheet":
                df_display = t_stmt.get('balance_sheet', pd.DataFrame())
            else:
                df_display = t_stmt.get('cash_flow', pd.DataFrame())

            if not df_display.empty:
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info(f"No statement data available for {selected_stmt_ticker}.")

    # ----------------------------------------------------
    # TAB 2: CREDIT WORTHINESS & SOLVENCY (NEW TAB)
    # ----------------------------------------------------
    with tab2:
        st.subheader("💳 Credit Worthiness, Debt Serviceability & Solvency")
        st.markdown("Evaluated by **Agent 2 (Fundamental & Credit Analyst)** assessing debt service capacity, leverage burden, and cash flow buffer under distress.")

        st.info("ℹ️ **Pure Financial Credit Metrics**: Evaluates reported leverage (Net Debt / EBITDA), coverage multiples (EBITDA / Interest Expense, FCCR), and cash flow solvency directly from SEC filings. Official agency ratings (S&P, Moody's, Fitch) are copyright-protected qualitative opinions issued by analyst committees and are not available via free public market feeds.")

        # Summary Table of Key Credit Ratios
        credit_table_data = []
        for t in tickers:
            cm = credit_res.get(t, {})
            tot_debt = f"${cm.get('total_debt', 0)/1e9:.2f}B" if not pd.isna(cm.get('total_debt')) else "N/A"
            net_d = cm.get('net_debt', np.nan)
            if pd.isna(net_d):
                net_debt_str = "N/A"
            elif net_d < 0:
                net_debt_str = f"-${abs(net_d)/1e9:.2f}B (Net Cash)"
            else:
                net_debt_str = f"${net_d/1e9:.2f}B"

            # Leverage
            td_ebitda = f"{cm.get('total_debt_to_ebitda'):.2f}x" if not pd.isna(cm.get('total_debt_to_ebitda')) else "N/A"
            nd_ebitda = cm.get('net_debt_to_ebitda')
            if pd.isna(nd_ebitda):
                nd_ebitda_str = "N/A"
            elif nd_ebitda < 0:
                nd_ebitda_str = f"{nd_ebitda:.2f}x (Net Cash)"
            else:
                nd_ebitda_str = f"{nd_ebitda:.2f}x"

            # Coverage
            ebitda_cov = cm.get('ebitda_interest_coverage')
            if pd.isna(ebitda_cov):
                ebitda_cov_str = "N/A"
            elif ebitda_cov >= 50:
                ebitda_cov_str = ">50.0x (Prime Buffer)"
            else:
                ebitda_cov_str = f"{ebitda_cov:.2f}x"

            fccr_val = cm.get('fccr')
            if pd.isna(fccr_val):
                fccr_str = "N/A"
            elif fccr_val >= 50:
                fccr_str = ">50.0x (Prime Buffer)"
            else:
                fccr_str = f"{fccr_val:.2f}x"

            # Liquidity / Solvency
            cfo_td = cm.get('cfo_to_total_debt')
            if pd.isna(cfo_td):
                cfo_td_str = "N/A"
            elif cfo_td >= 10:
                cfo_td_str = ">100% (Cash Flow >> Debt)"
            else:
                cfo_td_str = f"{cfo_td*100:.1f}%"

            qr = f"{cm.get('quick_ratio'):.2f}" if not pd.isna(cm.get('quick_ratio')) else "N/A"
            flags = ", ".join(cm.get('credit_flags', [])) if cm.get('credit_flags') else "Standard"

            # LTV & Capital Structure Multiples
            ltv_val = cm.get('ltv')
            ltv_str = f"{ltv_val*100:.1f}%" if not pd.isna(ltv_val) else "N/A"

            da_val = cm.get('debt_to_assets')
            da_str = f"{da_val*100:.1f}%" if not pd.isna(da_val) else "N/A"

            dc_val = cm.get('debt_to_capital')
            dc_str = f"{dc_val*100:.1f}%" if not pd.isna(dc_val) else "N/A"

            fl_val = cm.get('financial_leverage')
            fl_str = f"{fl_val:.2f}x" if not pd.isna(fl_val) else "N/A"

            credit_table_data.append({
                "Ticker": t,
                "Total Debt": tot_debt,
                "Net Debt": net_debt_str,
                "Market LTV [<20%]": ltv_str,
                "Debt / Assets [<30%]": da_str,
                "Debt / Capital [<35%]": dc_str,
                "Fin. Leverage [<3.0x]": fl_str,
                "Total Debt / EBITDA [<2.5x]": td_ebitda,
                "Net Debt / EBITDA [<1.5x]": nd_ebitda_str,
                "EBITDA / Int Exp [>8.0x]": ebitda_cov_str,
                "FCCR [>2.5x]": fccr_str,
                "CFO / Total Debt [>30%]": cfo_td_str,
                "Quick Ratio [>1.0x]": qr,
                "Credit Signals": flags
            })

        credit_df = pd.DataFrame(credit_table_data)
        st.markdown(render_styled_table(credit_df, wrap_cols=["Credit Signals"], badge_cols=["Credit Signals"]), unsafe_allow_html=True)

        st.markdown("---")

        # Comparative Credit & Leverage Charts
        st.markdown("### 📊 Comprehensive Credit & Leverage Profiles")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### ⚖️ Cash Flow Leverage: Total Debt vs. Net Debt to EBITDA")
            lev_data = []
            for t in tickers:
                cm = credit_res.get(t, {})
                td_eb = cm.get('total_debt_to_ebitda', 0) or 0
                nd_eb = cm.get('net_debt_to_ebitda', 0) or 0
                lev_data.append({"Ticker": t, "Metric": "Total Debt / EBITDA", "Value": max(td_eb, 0)})
                lev_data.append({"Ticker": t, "Metric": "Net Debt / EBITDA", "Value": nd_eb})

            lev_df = pd.DataFrame(lev_data)
            fig_lev = px.bar(
                lev_df, x="Ticker", y="Value", color="Metric", barmode="group",
                title="Leverage Multiples (Lower is Safer; <2.0x is IG)",
                labels={"Value": "Multiple (x)"},
                color_discrete_map={"Total Debt / EBITDA": "#E53E3E", "Net Debt / EBITDA": "#3182CE"}
            )
            fig_lev = lock_chart_for_mobile(fig_lev)
            st.plotly_chart(fig_lev, use_container_width=True, config=PLOTLY_CONFIG)

        with col_c2:
            st.markdown("#### 🛡️ Debt Service Coverage: Interest Coverage vs. FCCR")
            cov_data = []
            for t in tickers:
                cm = credit_res.get(t, {})
                eb_cov = min(cm.get('ebitda_interest_coverage', 0) or 0, 30.0)  # clamp for chart readability
                fccr_v = min(cm.get('fccr', 0) or 0, 30.0)
                cov_data.append({"Ticker": t, "Metric": "EBITDA Interest Coverage", "Value": max(eb_cov, 0)})
                cov_data.append({"Ticker": t, "Metric": "FCCR (Fixed Charge Coverage)", "Value": max(fccr_v, 0)})

            cov_df = pd.DataFrame(cov_data)
            fig_cov = px.bar(
                cov_df, x="Ticker", y="Value", color="Metric", barmode="group",
                title="Debt Coverage Buffer (Higher is Safer; >2.5x is Robust)",
                labels={"Value": "Coverage Multiple (x)"},
                color_discrete_map={"EBITDA Interest Coverage": "#805AD5", "FCCR (Fixed Charge Coverage)": "#38A169"}
            )
            fig_cov = lock_chart_for_mobile(fig_cov)
            st.plotly_chart(fig_cov, use_container_width=True, config=PLOTLY_CONFIG)

        col_c3, col_c4 = st.columns(2)

        with col_c3:
            st.markdown("#### 🏛️ Asset Cushion: Market LTV (%) vs. Debt-to-Assets (Book LTV %)")
            ltv_data = []
            for t in tickers:
                cm = credit_res.get(t, {})
                ltv_pct = (cm.get('ltv', 0) or 0) * 100
                da_pct = (cm.get('debt_to_assets', 0) or 0) * 100
                ltv_data.append({"Ticker": t, "Metric": "Market LTV (%)", "Value": max(ltv_pct, 0)})
                ltv_data.append({"Ticker": t, "Metric": "Debt / Assets (%)", "Value": max(da_pct, 0)})

            ltv_df = pd.DataFrame(ltv_data)
            fig_ltv = px.bar(
                ltv_df, x="Ticker", y="Value", color="Metric", barmode="group",
                title="Loan-to-Value & Asset Gearing (Lower is Safer; <25% is Fortress)",
                labels={"Value": "Percentage (%)"},
                color_discrete_map={"Market LTV (%)": "#00B4D8", "Debt / Assets (%)": "#7209B7"}
            )
            fig_ltv = lock_chart_for_mobile(fig_ltv)
            st.plotly_chart(fig_ltv, use_container_width=True, config=PLOTLY_CONFIG)

        with col_c4:
            st.markdown("#### 🏗️ Capital Structure Gearing: Debt-to-Capital (%)")
            cap_data = []
            for t in tickers:
                cm = credit_res.get(t, {})
                dc_pct = (cm.get('debt_to_capital', 0) or 0) * 100
                cap_data.append({"Ticker": t, "Debt / Capital (%)": max(dc_pct, 0)})

            cap_df = pd.DataFrame(cap_data)
            fig_cap = px.bar(
                cap_df, x="Ticker", y="Debt / Capital (%)",
                title="Debt-to-Capitalization (Debt / [Debt + Equity]; <35% is Conservative)",
                labels={"Debt / Capital (%)": "Debt / Capital (%)"},
                color="Debt / Capital (%)",
                color_continuous_scale="Teal"
            )
            fig_cap = lock_chart_for_mobile(fig_cap)
            st.plotly_chart(fig_cap, use_container_width=True, config=PLOTLY_CONFIG)

        # Institutional Analyst Callout Boxes: FCCR & LTV / Capital Structure
        col_box1, col_box2 = st.columns(2)

        with col_box1:
            st.markdown("""
            <div class="credit-box">
                <h4 style="margin-top:0; color:#38BDF8;">📘 Corporate LTV & Capital Structure Metrics</h4>
                <p>Institutional lenders and syndicated loan desks assess solvency beyond earnings through structural balance-sheet coverage:</p>
                <ul>
                    <li><b>Market LTV (Total Debt / Enterprise Value):</b> Measures the percentage of total corporate enterprise value encumbered by debt. An LTV &lt; 20% indicates massive equity cushion and superior recovery prospects.</li>
                    <li><b>Book LTV (Debt-to-Assets):</b> The balance sheet liquidation perspective: what share of total physical and intangible assets is claimed by creditors.</li>
                    <li><b>Debt-to-Capitalization:</b> Measures the permanent capital mix ($Debt / [Debt + Equity]$), reflecting financial leverage risk.</li>
                    <li><b>Financial Leverage Multiplier:</b> The DuPont asset multiplier ($Assets / Equity$).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_box2:
            st.markdown("""
            <div class="credit-box">
                <h4 style="margin-top:0; color:#38BDF8;">📘 Debt Service & Fixed Charge Coverage (FCCR)</h4>
                <p>While <b>EBITDA / Interest</b> is widely quoted, institutional credit rating agencies (S&P, Moody's) prefer <b>FCCR</b> for underwriting:</p>
                <ul>
                    <li><b>EBITDA Ignores Mandatory Outflows:</b> EBITDA excludes required <i>Cash Taxes</i>, <i>Maintenance CapEx</i>, and working capital needs.</li>
                    <li><b>Interest Coverage Ignores Non-Interest Charges:</b> Ignores contractual <i>Lease Payments (rent)</i> and <i>Mandatory Principal Repayments</i>.</li>
                    <li><i>Note on Growth CapEx:</i> Reported CapEx includes discretionary growth investments (e.g. AWS AI data centers) that can be curtailed during liquidity crunches.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Mathematical Definition of Fixed Charge Coverage Ratio (FCCR):")
        st.latex(r"""
        \text{FCCR} = \frac{\text{EBITDA} - \text{Maintenance CapEx} - \text{Cash Taxes}}{\text{Interest Expense} + \text{Mandatory Principal Repayments} + \text{Lease Payments}}
        """)

        # Key Credit Benchmarks Guide
        st.markdown("### 🎯 Institutional Credit Underwriting Benchmarks")
        st.caption("Standard institutional solvency ranges, formulas, and target thresholds used by corporate credit risk analysts:")

        credit_benchmark_rows = [
            {
                "Credit Ratio": "Market LTV",
                "Formula": "Total Debt / Enterprise Value",
                "🟢 Prime / Fortress": "< 20%",
                "🟡 Moderate / IG": "20% - 40%",
                "🔴 High Risk / Caution": "> 50%",
                "Underwriting Purpose": "Share of enterprise value encumbered by debt; lower provides superior equity cushion for lenders."
            },
            {
                "Credit Ratio": "Book LTV (Debt / Assets)",
                "Formula": "Total Debt / Total Assets",
                "🟢 Prime / Fortress": "< 25%",
                "🟡 Moderate / IG": "25% - 45%",
                "🔴 High Risk / Caution": "> 60%",
                "Underwriting Purpose": "Balance sheet liquidation perspective; percentage of total corporate assets claimed by creditors."
            },
            {
                "Credit Ratio": "Debt-to-Capitalization",
                "Formula": "Total Debt / (Total Debt + Equity)",
                "🟢 Prime / Fortress": "< 30%",
                "🟡 Moderate / IG": "30% - 45%",
                "🔴 High Risk / Caution": "> 55%",
                "Underwriting Purpose": "Measures permanent capital structure gearing; proportion of firm capital financed via debt."
            },
            {
                "Credit Ratio": "Financial Leverage Multiplier",
                "Formula": "Total Assets / Stockholders' Equity",
                "🟢 Prime / Fortress": "< 2.0x",
                "🟡 Moderate / IG": "2.0x - 3.5x",
                "🔴 High Risk / Caution": "> 4.5x or Deficit",
                "Underwriting Purpose": "DuPont asset multiplier; magnifies equity returns but amplifies insolvency vulnerability."
            },
            {
                "Credit Ratio": "Total Debt / EBITDA",
                "Formula": "Total Debt / EBITDA",
                "🟢 Prime / Fortress": "< 2.0x",
                "🟡 Moderate / IG": "2.0x - 3.5x",
                "🔴 High Risk / Caution": "> 4.5x",
                "Underwriting Purpose": "Gross leverage multiple; years of pre-tax cash flow required to extinguish all debt obligations."
            },
            {
                "Credit Ratio": "Net Debt / EBITDA",
                "Formula": "(Total Debt - Cash & Equiv.) / EBITDA",
                "🟢 Prime / Fortress": "< 1.5x (or Net Cash)",
                "🟡 Moderate / IG": "1.5x - 3.0x",
                "🔴 High Risk / Caution": "> 3.5x",
                "Underwriting Purpose": "Core syndicated loan covenant; true net leverage assuming liquid cash immediately pays down debt."
            },
            {
                "Credit Ratio": "EBITDA Interest Coverage",
                "Formula": "EBITDA / Interest Expense",
                "🟢 Prime / Fortress": "> 8.0x",
                "🟡 Moderate / IG": "3.0x - 8.0x",
                "🔴 High Risk / Caution": "< 2.5x",
                "Underwriting Purpose": "Earnings headroom relative to contractual interest obligations; vulnerability to rate hikes."
            },
            {
                "Credit Ratio": "Fixed Charge Coverage (FCCR)",
                "Formula": "(EBITDA - Maint CapEx - Tax) / (Int + Principal + Leases)",
                "🟢 Prime / Fortress": "> 3.0x",
                "🟡 Moderate / IG": "1.5x - 3.0x",
                "🔴 High Risk / Caution": "< 1.2x",
                "Underwriting Purpose": "Comprehensive cash coverage after taxes and essential capex; gold standard for loan underwriting."
            },
            {
                "Credit Ratio": "CFO / Total Debt",
                "Formula": "Operating Cash Flow / Total Debt",
                "🟢 Prime / Fortress": "> 35%",
                "🟡 Moderate / IG": "15% - 35%",
                "🔴 High Risk / Caution": "< 15%",
                "Underwriting Purpose": "Annual operating cash generated relative to total debt; core rating agency cash flow adequacy test."
            },
            {
                "Credit Ratio": "Quick Ratio (Acid Test)",
                "Formula": "(Cash + ST Inv + Receivables) / Current Liabilities",
                "🟢 Prime / Fortress": "> 1.0x",
                "🟡 Moderate / IG": "0.8x - 1.0x",
                "🔴 High Risk / Caution": "< 0.7x",
                "Underwriting Purpose": "Immediate liquidity buffer; ability to extinguish near-term debt without liquidating inventory."
            }
        ]
        credit_bench_df = pd.DataFrame(credit_benchmark_rows)
        st.markdown(render_styled_table(credit_bench_df, wrap_cols=["Underwriting Purpose"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        bench_col1, bench_col2, bench_col3 = st.columns(3)
        with bench_col1:
            st.success("**Fortress Balance Sheet & High Liquidity**\n- Market LTV: $< 20\\%$\n- Debt / Assets: $< 30\\%$\n- Net Debt / EBITDA: $< 1.5x$\n- EBITDA Interest Coverage: $> 8.0x$\n- CFO / Total Debt: $> 30\\%$\n- *Substantial liquidity reserves; minimal debt service risk.*")
        with bench_col2:
            st.warning("**Moderate Leverage & Adequate Capacity**\n- Market LTV: $20\\% - 40\\%$\n- Debt / Assets: $30\\% - 50\\%$\n- Net Debt / EBITDA: $1.5x - 3.5x$\n- EBITDA Interest Coverage: $3.0x - 8.0x$\n- CFO / Total Debt: $15\\% - 30\\%$\n- *Adequate debt service capability; monitored during economic downturns.*")
        with bench_col3:
            st.error("**Elevated Leverage & Refinancing Exposure**\n- Market LTV: $> 50\\%$\n- Debt / Assets: $> 60\\%$\n- Net Debt / EBITDA: $> 4.0x$\n- EBITDA Interest Coverage: $< 2.5x$\n- CFO / Total Debt: $< 15\\%$\n- *High sensitivity to interest rates, refinancing & debt maturity burden.*")

    # ----------------------------------------------------
    # TAB 3: HISTORICAL PERFORMANCE & RISK
    # ----------------------------------------------------
    with tab3:
        st.subheader("Historical Stock Performance & Institutional Risk Metrics")
        st.markdown(f"Ingested by **Agent 1 (Data Architect)** and processed by **Agent 3 (Quant Analyst)** (Optimization Mode: **{'🤖 ML Expected Returns' if use_ml_views else '📊 Historical Mean Returns'}**).")

        # Rebased Cumulative Returns Plot
        normalized_prices = (prices_df / prices_df.iloc[0]) * 100
        fig_price = px.line(
            normalized_prices,
            x=normalized_prices.index,
            y=normalized_prices.columns,
            title="Rebased Asset Growth (Initial $100 Baseline)",
            labels={'value': 'Rebased Price ($)', 'variable': 'Ticker', 'Date': 'Date'}
        )
        fig_price = lock_chart_for_mobile(fig_price)
        st.plotly_chart(fig_price, use_container_width=True, config=PLOTLY_CONFIG)

        # Risk Metrics Summary Table (with VaR and CVaR)
        asset_m = quant_res['asset_metrics']
        risk_table_data = []
        for t in tickers:
            am = asset_m.get(t, {})
            risk_table_data.append({
                "Ticker": t,
                "CAGR": f"{am.get('cagr', 0)*100:.2f}%",
                "Expected Return": f"{am.get('annualized_return', 0)*100:.2f}%",
                "Annualized Volatility": f"{am.get('annualized_volatility', 0)*100:.2f}%",
                "Sharpe Ratio": f"{am.get('sharpe_ratio', 0):.2f}",
                "Max Drawdown": f"{am.get('max_drawdown', 0)*100:.2f}%",
                "95% VaR (Ann.)": f"{am.get('var_95', 0)*100:.2f}%",
                "95% CVaR / Expected Shortfall": f"{am.get('cvar_95', 0)*100:.2f}%"
            })

        st.markdown("### Institutional Risk & Return Summary Table")
        st.dataframe(pd.DataFrame(risk_table_data), use_container_width=True, hide_index=True)

        # Drawdown Underwater Plot
        st.markdown("### Historical Drawdown (Underwater Chart)")
        returns_df = quant_res['returns_df']
        if not returns_df.empty:
            cum_returns = (1 + returns_df).cumprod()
            peak = cum_returns.cummax()
            drawdowns = (cum_returns - peak) / peak
            fig_dd = px.line(
                drawdowns * 100,
                x=drawdowns.index,
                y=drawdowns.columns,
                title="Historical Drawdowns (% from Peak)",
                labels={'value': 'Drawdown (%)', 'variable': 'Ticker', 'Date': 'Date'}
            )
            fig_dd = lock_chart_for_mobile(fig_dd)
            st.plotly_chart(fig_dd, use_container_width=True, config=PLOTLY_CONFIG)

        # Correlation Heatmap
        st.markdown("### Cross-Asset Return Correlation Matrix")
        corr_df = pd.DataFrame(quant_res['correlation_matrix'])
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Cross-Asset Daily Return Correlations"
        )
        fig_corr = lock_chart_for_mobile(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

    # ----------------------------------------------------
    # TAB 4: PREDICTIVE ML RETURN FORECASTING (AGENT 4)
    # ----------------------------------------------------
    with tab4:
        st.subheader("Predictive Machine Learning Return Forecasting")
        st.markdown(f"Engineered by **Agent 4 (Predictive ML Analyst)** using a **{ml_res.get('model_used', 'Random Forest Regressor')}** trained on technical features (14d RSI, 20d Volatility, 20d & 50d Momentum, Volume Trend).")

        ml_dict = ml_res.get('ml_results', {})
        latest_feats = ml_res.get('latest_features', {})

        # Section A: Live Technical Features Display Table
        st.markdown("### 📈 Live Technical Feature Inputs")
        feat_rows = []
        for t in tickers:
            f = latest_feats.get(t, {})
            feat_rows.append({
                "Ticker": t,
                "14-Day RSI": f"{f.get('rsi_14', 0):.2f}",
                "20-Day Volatility (Ann.)": f"{f.get('volatility_20d', 0)*100:.2f}%",
                "20-Day Momentum": f"{f.get('momentum_20d', 0)*100:+.2f}%",
                "50-Day Momentum": f"{f.get('momentum_50d', 0)*100:+.2f}%",
                "Volume Trend Ratio": f"{f.get('volume_trend', 0):.2f}",
                "SMA (20d / 50d) Ratio": f"{f.get('sma_ratio', 0):.2f}"
            })

        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
        st.markdown("---")

        # Section B: Supervised ML Forecasts Table & Bar Chart
        st.markdown("### 🔮 Supervised ML Forward Return Forecasts")
        ml_summary_rows = []
        for t in tickers:
            t_ml = ml_dict.get(t, {})
            ml_summary_rows.append({
                "Ticker": t,
                "Predicted 20-Day Return": f"{t_ml.get('predicted_20d_return', 0)*100:+.2f}%",
                "Forecasted Annualized Return": f"{t_ml.get('predicted_annualized_return', 0)*100:+.2f}%",
                "Test Holdout Directional Accuracy": f"{t_ml.get('directional_accuracy_pct', 0):.1f}%",
                "Model MAE": f"{t_ml.get('mae', 0):.4f}"
            })

        st.dataframe(pd.DataFrame(ml_summary_rows), use_container_width=True, hide_index=True)

        # Forecast Comparison Bar Chart & Feature Importance
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            pred_20d = [ml_dict.get(t, {}).get('predicted_20d_return', 0)*100 for t in tickers]
            fig_ml_bar = px.bar(
                x=tickers, y=pred_20d,
                labels={'x': 'Ticker', 'y': 'Predicted 20d Return (%)'},
                title="ML 20-Day Forward Return Forecast (%)",
                color=pred_20d,
                color_continuous_scale="Viridis"
            )
            fig_ml_bar = lock_chart_for_mobile(fig_ml_bar)
            st.plotly_chart(fig_ml_bar, use_container_width=True, config=PLOTLY_CONFIG)

        with col_ml2:
            st.markdown("### 🌲 Feature Importance Breakdown")
            sel_ml_ticker = st.selectbox("Select Ticker for Feature Importance Analysis", tickers)
            if sel_ml_ticker in ml_dict:
                fi_dict = ml_dict[sel_ml_ticker].get('feature_importances', {})
                fi_df = pd.DataFrame(list(fi_dict.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
                fig_fi = px.bar(
                    fi_df, x='Importance', y='Feature', orientation='h',
                    title=f"Random Forest Feature Importance for {sel_ml_ticker}",
                    color='Importance', color_continuous_scale="Blues"
                )
                fig_fi = lock_chart_for_mobile(fig_fi)
                st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)

    # ----------------------------------------------------
    # TAB 5: PORTFOLIO OPTIMIZATION & VaR
    # ----------------------------------------------------
    with tab5:
        active_cap_str = f"{quant_res.get('max_asset_weight', 1.0)*100:.0f}% per asset" if quant_res.get('max_asset_weight', 1.0) < 0.99 else "Unconstrained (100%)"
        st.markdown(f"Calculated by **Agent 3 (Quantitative Analyst)** under **$R_f$ = {rf_rate_pct:.2f}%**, **Return Mode = {'🤖 Random Forest ML Forecasts' if use_ml_views else '📊 Empirical Historical Returns'}**, and **Position Cap = {active_cap_str}**.")

        max_sharpe = quant_res['max_sharpe_portfolio']
        min_var = quant_res['min_variance_portfolio']
        mc = quant_res['monte_carlo']
        ef = quant_res['efficient_frontier']

        # Highlight Metric Cards: Return, Volatility, Sharpe & Asymmetric Downside Risk
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Max Sharpe Expected Return", f"{max_sharpe['expected_return']*100:.2f}%")
        with col_m2:
            st.metric("Max Sharpe Volatility", f"{max_sharpe['volatility']*100:.2f}%")
        with col_m3:
            st.metric("Max Sharpe Ratio", f"{max_sharpe['sharpe_ratio']:.2f}")
        with col_m4:
            st.metric("Sortino Ratio (Downside)", f"{max_sharpe.get('sortino_ratio', 0):.2f}", help="Excess return divided by downside deviation below the risk-free rate.")

        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        with col_m5:
            st.metric("1-Day 95% VaR", f"{max_sharpe.get('var_95_1d', 0)*100:.2f}%", help="Empirical 1-day 95% Value at Risk.")
        with col_m6:
            st.metric("1-Year 95% VaR (CF)", f"{max_sharpe.get('var_95', 0)*100:.2f}%", help="Annualized 95% Value at Risk via Cornish-Fisher expansion.")
        with col_m7:
            st.metric("Historical Max Drawdown", f"{max_sharpe.get('max_drawdown', 0)*100:.2f}%", help="Peak-to-trough historical drawdown of the synthesized portfolio.")
        with col_m8:
            lw_int = quant_res.get('shrinkage_intensity', 0.0)
            st.metric("Ledoit-Wolf Shrinkage (δ*)", f"{lw_int:.3f}", help="Optimal shrinkage intensity regularizing the sample covariance matrix against noise.")

        if max_sharpe['expected_return'] > 0.25 or max_sharpe['volatility'] > 0.30:
            st.warning(
                f"⚠️ **Institutional Sanity Check — High Volatility & Historical Backtest Distortion**\n\n"
                f"- **Absurdly High Expected Return ({max_sharpe['expected_return']*100:.2f}%) vs. Benchmark Norms:** Standard broad equity benchmarks (e.g., S&P 500) average roughly 10%–12% annualized returns with 15%–18% volatility. A projected return >25% mathematically occurs only when one or more underlying assets experienced explosive gains during the historical test period (e.g. crypto, leveraged tech, or high-beta momentum stocks).\n"
                f"- **'Empirical Historical Returns' Rearview-Mirror Bias:** In empirical return mode, the optimizer linearly extrapolates past extraordinary bull runs into the future without mean-reversion dampening or forward-looking Capital Market Assumptions (CMAs).\n"
                f"- **Extreme Volatility ({max_sharpe['volatility']*100:.2f}%) & Ordinary Sharpe ({max_sharpe['sharpe_ratio']:.2f}):** A Sharpe ratio around 1.0–1.2 is conventional, not extraordinary. The high absolute return is driven purely by taking on massive raw volatility rather than superior risk-adjusted alpha.\n"
                f"- **Severe Tail Risk (95% VaR: {max_sharpe.get('var_95', 0)*100:.2f}%):** In adverse market regimes (the worst 5% of annual outcomes), this asset mix faces severe drawdowns that could erase over half to two-thirds of portfolio capital."
            )

        st.markdown("---")

        # Efficient Frontier Plotly Scatter Plot with Capital Allocation Line (CAL)
        st.markdown("### Interactive Efficient Frontier & Capital Allocation Line (CAL)")

        fig_ef = go.Figure()

        # Monte Carlo Portfolios
        fig_ef.add_trace(go.Scatter(
            x=[v * 100 for v in mc['volatilities']],
            y=[r * 100 for r in mc['returns']],
            mode='markers',
            marker=dict(
                size=5,
                color=mc['sharpe_ratios'],
                colorscale='Viridis',
                colorbar=dict(title=f"Sharpe"),
                showscale=True
            ),
            name="Simulated Portfolios",
            hoverinfo='text',
            text=[f"Return: {r*100:.2f}%<br>Vol: {v*100:.2f}%<br>Sharpe: {s:.2f}" 
                  for r, v, s in zip(mc['returns'], mc['volatilities'], mc['sharpe_ratios'])]
        ))

        # Efficient Frontier Line
        ef_vols = [v * 100 for v in ef['volatilities'] if not pd.isna(v)]
        ef_rets = [r * 100 for r, v in zip(ef['target_returns'], ef['volatilities']) if not pd.isna(v)]
        fig_ef.add_trace(go.Scatter(
            x=ef_vols,
            y=ef_rets,
            mode='lines',
            line=dict(color='orange', width=3, dash='dash'),
            name="Efficient Frontier"
        ))

        # Capital Allocation Line (CAL) from Rf to Max Sharpe
        if max_sharpe['volatility'] > 0:
            cal_x = [0.0, max_sharpe['volatility'] * 150]
            cal_y = [rf_rate * 100, (rf_rate + 1.5 * (max_sharpe['expected_return'] - rf_rate)) * 100]
            fig_ef.add_trace(go.Scatter(
                x=cal_x,
                y=cal_y,
                mode='lines',
                line=dict(color='rgba(0, 230, 118, 0.7)', width=2, dash='dot'),
                name="Capital Allocation Line (CAL)"
            ))

        # Highlight Max Sharpe Portfolio
        fig_ef.add_trace(go.Scatter(
            x=[max_sharpe['volatility'] * 100],
            y=[max_sharpe['expected_return'] * 100],
            mode='markers',
            marker=dict(color='red', size=14, symbol='star'),
            name="Max Sharpe",
            text=f"Max Sharpe<br>Return: {max_sharpe['expected_return']*100:.2f}%<br>Vol: {max_sharpe['volatility']*100:.2f}%<br>Sharpe: {max_sharpe['sharpe_ratio']:.2f}"
        ))

        # Highlight Min Variance Portfolio
        fig_ef.add_trace(go.Scatter(
            x=[min_var['volatility'] * 100],
            y=[min_var['expected_return'] * 100],
            mode='markers',
            marker=dict(color='cyan', size=12, symbol='diamond'),
            name="Min Variance",
            text=f"Min Variance<br>Return: {min_var['expected_return']*100:.2f}%<br>Vol: {min_var['volatility']*100:.2f}%"
        ))

        fig_ef.update_layout(
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Expected Return (%)",
            hovermode="closest",
            margin=dict(l=10, r=10, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_ef = lock_chart_for_mobile(fig_ef)
        st.plotly_chart(fig_ef, use_container_width=True, config=PLOTLY_CONFIG)

        # Optimal Allocations Pie Charts
        st.markdown("### Optimal Portfolio Allocation Breakdown")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            ms_w = max_sharpe['weights']
            fig_ms_pie = px.pie(
                names=list(ms_w.keys()),
                values=[v * 100 for v in ms_w.values()],
                title=f"Max Sharpe Allocation (Sharpe = {max_sharpe['sharpe_ratio']:.2f})",
                hole=0.4
            )
            fig_ms_pie = lock_chart_for_mobile(fig_ms_pie)
            st.plotly_chart(fig_ms_pie, use_container_width=True, config=PLOTLY_CONFIG)

        with col_p2:
            mv_w = min_var['weights']
            fig_mv_pie = px.pie(
                names=list(mv_w.keys()),
                values=[v * 100 for v in mv_w.values()],
                title=f"Minimum Variance Allocation (Vol = {min_var['volatility']*100:.2f}%)",
                hole=0.4
            )
            fig_mv_pie = lock_chart_for_mobile(fig_mv_pie)
            st.plotly_chart(fig_mv_pie, use_container_width=True, config=PLOTLY_CONFIG)

        # Allocation Weights Table
        alloc_df = pd.DataFrame({
            "Ticker": tickers,
            "Max Sharpe Weight (%)": [f"{max_sharpe['weights'].get(t, 0)*100:.2f}%" for t in tickers],
            "Min Variance Weight (%)": [f"{min_var['weights'].get(t, 0)*100:.2f}%" for t in tickers]
        })
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Interactive Portfolio Rebalancing Simulator
        st.subheader("🎛️ Custom Allocation Simulator & Rebalancer")
        st.markdown("Adjust asset weights manually to evaluate expected portfolio metrics against MPT optimal targets.")

        user_weights = {}
        cols_slider = st.columns(min(len(tickers), 5))
        default_w = 100.0 / len(tickers)

        for i, t in enumerate(tickers):
            with cols_slider[i % len(cols_slider)]:
                user_weights[t] = st.slider(f"{t} Weight (%)", 0.0, 100.0, default_w, step=1.0)

        total_w = sum(user_weights.values())

        if abs(total_w - 100.0) > 0.1:
            st.warning(f"Total Portfolio Weight is **{total_w:.1f}%**. Please adjust sliders to equal exactly 100.0%.")
        else:
            w_vec = np.array([user_weights[t] / 100.0 for t in tickers])
            ret_series = quant_res['returns_df']
            mean_rets = np.array(quant_res['mean_returns'])
            cov_mat = ret_series.cov() * 252

            custom_ret = np.sum(mean_rets * w_vec)
            custom_vol = np.sqrt(np.dot(w_vec.T, np.dot(cov_mat, w_vec)))
            custom_sharpe = (custom_ret - rf_rate) / custom_vol if custom_vol > 0 else 0

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                st.metric("Custom Expected Return", f"{custom_ret*100:.2f}%")
            with sim_c2:
                st.metric("Custom Volatility", f"{custom_vol*100:.2f}%")
            with sim_c3:
                st.metric("Custom Sharpe Ratio", f"{custom_sharpe:.2f}")

        # Export Allocation JSON
        st.markdown("### Export Portfolio Configuration")
        export_bundle = {
            "tickers": tickers,
            "risk_free_rate": rf_rate,
            "return_multiplier": ret_multiplier,
            "use_ml_views": use_ml_views,
            "max_sharpe_weights": max_sharpe['weights'],
            "min_variance_weights": min_var['weights'],
            "user_custom_weights": user_weights if abs(total_w - 100.0) <= 0.1 else "Invalid"
        }

        st.download_button(
            label="📥 Download Portfolio JSON Config",
            data=json.dumps(export_bundle, indent=2),
            file_name="portfolio_allocation.json",
            mime="application/json"
        )

        st.markdown("---")
        st.subheader("🌪️ Institutional Macro Scenario Stress-Testing")
        st.markdown("Simulates projected peak-to-trough drawdowns for the **Max Sharpe** portfolio under historical systemic market shocks:")

        stress_data = quant_res.get('stress_tests', [])
        if stress_data:
            stress_df = pd.DataFrame(stress_data)
            st.dataframe(
                stress_df[['Scenario', 'Historical Period', 'Benchmark Shock', 'Simulated Portfolio Loss', 'Context']],
                use_container_width=True,
                hide_index=True
            )

    # ----------------------------------------------------
    # TAB 6: GEMINI AI EXECUTIVE REPORT (OPTIONAL)
    # ----------------------------------------------------
    with tab6:
        st.subheader("📝 Gemini AI Executive Investment Report")
        st.markdown("Generated by **Agent 5 (Gemini AI Executive Summarizer)** synthesizing outputs across all 4 analytical agents.")

        report_text = ai_report_res.get('report', 'No report generated.')
        st.markdown(report_text)

if __name__ == "__main__":
    main()
