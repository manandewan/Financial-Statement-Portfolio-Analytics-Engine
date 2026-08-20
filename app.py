import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json

from src.agents.coordinator import AgentSystemCoordinator

# Page Configuration
st.set_page_config(
    page_title="Financial Statement & Portfolio Optimization Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark / Modern Aesthetic)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #A0AEC0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .agent-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .pill-data { background-color: #2B6CB0; color: white; }
    .pill-fund { background-color: #2F855A; color: white; }
    .pill-quant { background-color: #6B46C1; color: white; }
    .pill-ml { background-color: #D69E2E; color: black; }
    .pill-dev { background-color: #DD6B20; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def run_agent_pipeline(tickers_tuple, start_str, end_str, rf_rate):
    """
    Cached helper to execute multi-agent coordinator pipeline.
    """
    coordinator = AgentSystemCoordinator()
    return coordinator.run_pipeline(
        tickers=list(tickers_tuple),
        start_date=start_str,
        end_date=end_str,
        risk_free_rate=rf_rate
    )

def main():
    # Sidebar Configuration
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

    # Risk-free rate
    rf_rate_pct = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.25)
    rf_rate = rf_rate_pct / 100.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Multi-Agent Architecture")
    st.sidebar.markdown("""
    - **Agent 1: Data Architect**: Ingests prices & statements.
    - **Agent 2: Fundamental Analyst**: Computes D/E, ROE, FCF Yield.
    - **Agent 3: Quantitative Analyst**: MPT Optimization & Efficient Frontier.
    - **Agent 4: Predictive ML Analyst**: Random Forest return forecasts & feature inputs.
    - **Agent 5: Dashboard Developer**: Streamlit & Plotly UI.
    """)

    st.sidebar.button("🚀 Run Agent Pipeline", type="primary", use_container_width=True)

    # App Header
    st.markdown('<div class="main-header">Financial Statement & Portfolio Optimization Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent AI & Machine Learning Pipeline for Fundamental Valuation, Return Forecasting & Modern Portfolio Theory (MPT)</div>', unsafe_allow_html=True)

    # Pipeline Agent Badge Indicator
    st.markdown("""
    <div>
        <span class="agent-pill pill-data">Agent 1: Financial Data Architect</span>
        <span class="agent-pill pill-fund">Agent 2: Fundamental Analyst</span>
        <span class="agent-pill pill-quant">Agent 3: Quantitative Analyst</span>
        <span class="agent-pill pill-ml">Agent 4: Predictive ML Analyst</span>
        <span class="agent-pill pill-dev">Agent 5: Full-Stack Developer</span>
    </div>
    <br>
    """, unsafe_allow_html=True)

    if not ticker_list:
        st.warning("Please enter at least one stock ticker in the sidebar to run analysis.")
        st.stop()

    # Execution Trigger
    with st.spinner("🔄 Coordinating Agents: Ingesting data, analyzing fundamentals, training ML models, and optimizing portfolio..."):
        try:
            data_bundle = run_agent_pipeline(
                tickers_tuple=tuple(ticker_list),
                start_str=start_date.strftime("%Y-%m-%d"),
                end_str=end_date.strftime("%Y-%m-%d"),
                rf_rate=rf_rate
            )
        except Exception as e:
            st.error(f"Error running pipeline: {str(e)}")
            st.stop()

    # Unpack Bundle
    tickers = data_bundle['tickers']
    prices_df = data_bundle['prices']
    statements = data_bundle['statements']
    fundamental_res = data_bundle['fundamental']['metrics']
    ml_res = data_bundle.get('ml_predictive', {})
    quant_res = data_bundle['quant']

    # Tabs Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Fundamental Health (Agent 2)",
        "📊 Historical Performance",
        "🤖 ML Return Forecasting (Agent 4)",
        "🎯 Portfolio Optimization (Agent 3)"
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
                "Debt-to-Equity": de,
                "Current Ratio": cr,
                "Return on Equity (ROE)": roe,
                "FCF Yield": fcf_y,
                "Health Alerts": ", ".join(m.get('flags', [])) if m.get('flags') else "Normal"
            })

        fund_df = pd.DataFrame(fund_table_data)
        st.dataframe(fund_df, use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig_roe, use_container_width=True)

        with c2:
            de_vals = [fundamental_res.get(t, {}).get('debt_to_equity', 0) or 0 for t in tickers]
            fig_de = px.bar(
                x=tickers, y=de_vals,
                labels={'x': 'Ticker', 'y': 'D/E Ratio'},
                title="Debt-to-Equity Ratio",
                color=de_vals,
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_de, use_container_width=True)

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
    # TAB 2: HISTORICAL PERFORMANCE
    # ----------------------------------------------------
    with tab2:
        st.subheader("Historical Stock Performance & Risk Metrics")
        st.markdown("Ingested by **Agent 1 (Data Architect)** and processed by **Agent 3 (Quant Analyst)**.")

        # Rebased Cumulative Returns Plot
        normalized_prices = (prices_df / prices_df.iloc[0]) * 100
        fig_price = px.line(
            normalized_prices,
            x=normalized_prices.index,
            y=normalized_prices.columns,
            title="Rebased Asset Growth (Initial $100 Baseline)",
            labels={'value': 'Rebased Price ($)', 'variable': 'Ticker', 'Date': 'Date'}
        )
        fig_price.update_layout(hovermode="x unified")
        st.plotly_chart(fig_price, use_container_width=True)

        # Risk Metrics Summary Table
        asset_m = quant_res['asset_metrics']
        risk_table_data = []
        for t in tickers:
            am = asset_m.get(t, {})
            risk_table_data.append({
                "Ticker": t,
                "CAGR": f"{am.get('cagr', 0)*100:.2f}%",
                "Annualized Return": f"{am.get('annualized_return', 0)*100:.2f}%",
                "Annualized Volatility": f"{am.get('annualized_volatility', 0)*100:.2f}%",
                "Sharpe Ratio": f"{am.get('sharpe_ratio', 0):.2f}",
                "Max Drawdown": f"{am.get('max_drawdown', 0)*100:.2f}%"
            })

        st.markdown("### Individual Asset Risk & Return Summary")
        st.dataframe(pd.DataFrame(risk_table_data), use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_corr, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: PREDICTIVE ML RETURN FORECASTING (AGENT 4)
    # ----------------------------------------------------
    with tab3:
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
            st.plotly_chart(fig_ml_bar, use_container_width=True)

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
                st.plotly_chart(fig_fi, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: PORTFOLIO OPTIMIZATION
    # ----------------------------------------------------
    with tab4:
        st.subheader("Modern Portfolio Theory (MPT) Optimization")
        st.markdown("Calculated by **Agent 3 (Quantitative Analyst)** using `scipy.optimize` and Monte Carlo simulation.")

        max_sharpe = quant_res['max_sharpe_portfolio']
        min_var = quant_res['min_variance_portfolio']
        mc = quant_res['monte_carlo']
        ef = quant_res['efficient_frontier']

        # Highlight Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Max Sharpe Return", f"{max_sharpe['expected_return']*100:.2f}%")
        with col_m2:
            st.metric("Max Sharpe Volatility", f"{max_sharpe['volatility']*100:.2f}%")
        with col_m3:
            st.metric("Max Sharpe Ratio", f"{max_sharpe['sharpe_ratio']:.2f}")
        with col_m4:
            st.metric("Min Volatility", f"{min_var['volatility']*100:.2f}%")

        st.markdown("---")

        # Efficient Frontier Plotly Scatter Plot
        st.markdown("### Interactive Efficient Frontier Scatter Plot")

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
                colorbar=dict(title="Sharpe Ratio"),
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

        # Highlight Max Sharpe Portfolio
        fig_ef.add_trace(go.Scatter(
            x=[max_sharpe['volatility'] * 100],
            y=[max_sharpe['expected_return'] * 100],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name="Max Sharpe Ratio Portfolio",
            text=f"Max Sharpe<br>Return: {max_sharpe['expected_return']*100:.2f}%<br>Vol: {max_sharpe['volatility']*100:.2f}%"
        ))

        # Highlight Min Variance Portfolio
        fig_ef.add_trace(go.Scatter(
            x=[min_var['volatility'] * 100],
            y=[min_var['expected_return'] * 100],
            mode='markers',
            marker=dict(color='cyan', size=13, symbol='diamond'),
            name="Minimum Variance Portfolio",
            text=f"Min Variance<br>Return: {min_var['expected_return']*100:.2f}%<br>Vol: {min_var['volatility']*100:.2f}%"
        ))

        fig_ef.update_layout(
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Expected Return (%)",
            hovermode="closest",
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig_ef, use_container_width=True)

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
            st.plotly_chart(fig_ms_pie, use_container_width=True)

        with col_p2:
            mv_w = min_var['weights']
            fig_mv_pie = px.pie(
                names=list(mv_w.keys()),
                values=[v * 100 for v in mv_w.values()],
                title=f"Minimum Variance Allocation (Vol = {min_var['volatility']*100:.2f}%)",
                hole=0.4
            )
            st.plotly_chart(fig_mv_pie, use_container_width=True)

        # Allocation Weights Table
        alloc_df = pd.DataFrame({
            "Ticker": tickers,
            "Max Sharpe Weight (%)": [f"{max_sharpe['weights'].get(t, 0)*100:.2f}%" for t in tickers],
            "Min Variance Weight (%)": [f"{min_var['weights'].get(t, 0)*100:.2f}%" for t in tickers]
        })
        st.table(alloc_df)

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
            # Calculate custom portfolio metrics
            w_vec = np.array([user_weights[t] / 100.0 for t in tickers])
            ret_series = quant_res['returns_df']
            mean_rets = ret_series.mean() * 252
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

if __name__ == "__main__":
    main()
