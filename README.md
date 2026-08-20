# Financial Statement & Portfolio Analytics Engine

An interactive, multi-agent Financial Statement, Predictive Machine Learning, & Portfolio Optimization Dashboard built with Python, Streamlit, Plotly, Scikit-Learn, and Modern Portfolio Theory (MPT).

## 🌟 Overview

The **Financial Statement & Portfolio Analytics Engine** coordinates specialized AI & Quantitative agents to ingest financial statements, evaluate fundamental corporate health metrics, train Predictive Supervised Machine Learning models, execute MPT portfolio optimizations, and display interactive visual analytics.

---

## 🤖 Multi-Agent Architecture

```mermaid
flowchart TD
    A["User Inputs (Tickers, Dates, Risk-Free Rate)"] --> B["Agent 1: Financial Data Architect"]
    B -->|"Raw Prices & Statements"| C["Agent 2: Fundamental Analyst"]
    B -->|"Price & Feature History"| D["Agent 4: Predictive ML Analyst"]
    B -->|"Historical Price Series"| E["Agent 3: Quantitative Analyst"]
    C -->|"D/E, Current Ratio, ROE, FCF Yield"| F["Agent System Coordinator"]
    D -->|"Random Forest 20d Forward Return Forecasts"| F
    E -->|"Efficient Frontier & MPT Weights"| F
    F -->|"Compiled Analytics Data Bundle"| G["Agent 5: Dashboard Developer (app.py)"]
    G --> H["Interactive Web Interface"]
```

### Agent Roles

1. **Agent 1: Financial Data Architect (`src/agents/data_architect.py`)**
   - Fetches 5-year historical daily price data and financial statements (Income Statement, Balance Sheet, Cash Flow) via `yfinance`.
   - Cleans missing values, aligns reporting dates, and standardizes data schema.

2. **Agent 2: Fundamental Analyst (`src/agents/fundamental_analyst.py`)**
   - Extracts key corporate health metrics: **Debt-to-Equity**, **Current Ratio**, **Return on Equity (ROE)**, and **Free Cash Flow Yield**.
   - Generates automated health alerts and financial risk flags.

3. **Agent 3: Quantitative Analyst (`src/agents/quant_analyst.py`)**
   - Calculates CAGR, annualized return, volatility, max drawdown, and cross-asset return correlation matrices.
   - Solves MPT optimizations using `scipy.optimize`:
     - **Max Sharpe Ratio Portfolio**
     - **Minimum Variance Portfolio**
     - **Efficient Frontier Curve** & Monte Carlo portfolio simulation.

4. **Agent 4: Predictive Machine Learning Analyst (`src/agents/ml_predictive_analyst.py`)**
   - Constructs technical feature matrices (14-day RSI, rolling volatility, 20d/50d SMA ratios, price momentum).
   - Trains Supervised ML Regressors (**Random Forest** & **Ridge**) on 80/20 train-test chronological splits.
   - Forecasts 20-day forward return predictions and outputs feature importance analysis per ticker.

5. **Agent 5: Full-Stack Dashboard Developer (`app.py`)**
   - Renders a multi-tab Streamlit dashboard with Plotly visual analytics, financial statement inspector, ML return forecast visualizer, and custom portfolio rebalancing simulator.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- `pip`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manandewan/Financial-Statement-Portfolio-Analytics-Engine.git
   cd Financial-Statement-Portfolio-Analytics-Engine
   ```

2. **Set up virtual environment & install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   PYTHONPATH=. streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 🧪 Running Tests

To run the automated test suite:
```bash
source .venv/bin/activate
PYTHONPATH=. python -m unittest discover -s tests
```

---

## 📁 Repository Structure

```
├── app.py                      # Agent 5: Streamlit Dashboard Application
├── src/
│   ├── agents/
│   │   ├── data_architect.py    # Agent 1: Data Ingestion & Cleaning
│   │   ├── fundamental_analyst.py # Agent 2: Fundamental Ratios
│   │   ├── quant_analyst.py    # Agent 3: MPT Portfolio Optimizer
│   │   ├── ml_predictive_analyst.py # Agent 4: Supervised ML Return Forecaster
│   │   └── coordinator.py      # Master Agent Orchestrator
├── tests/
│   ├── test_agents.py          # Unit Test Suite
│   └── test_live_pipeline.py   # Live Pipeline Integration Test
├── requirements.txt            # Project Dependencies
├── README.md                   # System Documentation
└── .gitignore
```
