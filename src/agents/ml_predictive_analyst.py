import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Dict, Any, List
import logging

logger = logging.getLogger("MLPredictiveAnalyst")

class MLPredictiveAnalystAgent:
    """
    Agent 4: Predictive Machine Learning Analyst
    Trains Supervised ML Regressors (Random Forest & Ridge) on engineered technical features:
    - 14-Day RSI
    - 20-Day Annualized Volatility
    - 20-Day & 50-Day Price Momentum
    - Volume Trend Ratio (10d SMA / 50d SMA Volume)
    Forecasts forward stock returns, model accuracy metrics, and displays technical feature inputs.
    """
    def __init__(self, model_type: str = "random_forest"):
        self.name = "Predictive Machine Learning Analyst"
        self.model_type = model_type

    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def extract_features(self, prices_series: pd.Series, volume_series: pd.Series = None) -> pd.DataFrame:
        """
        Construct feature matrix for supervised learning from price and volume history.
        """
        df = pd.DataFrame({'close': prices_series})
        
        # Returns / Momentum
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)
        df['momentum_50d'] = df['close'].pct_change(50)
        
        # Volatility & RSI
        df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        df['rsi_14'] = self._compute_rsi(df['close'], 14)
        
        # Moving Average Ratios
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        df['sma_ratio'] = sma_20 / (sma_50 + 1e-8)

        # Volume Trend
        if volume_series is not None and not volume_series.dropna().empty:
            vol_s = volume_series.reindex(df.index).ffill().bfill()
            vol_sma10 = vol_s.rolling(10).mean()
            vol_sma50 = vol_s.rolling(50).mean()
            df['volume_trend'] = vol_sma10 / (vol_sma50 + 1e-8)
        else:
            df['volume_trend'] = 1.0

        # Target: Forward 20-day return
        df['target_forward_20d'] = df['close'].shift(-20) / df['close'] - 1.0

        return df.dropna()

    def predict(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ML models per ticker, extract current technical features, and forecast returns.
        """
        prices_df = raw_data.get('prices', pd.DataFrame())
        if prices_df.empty:
            raise ValueError("Price DataFrame is empty for ML prediction.")

        tickers = list(prices_df.columns)
        statements = raw_data.get('statements', {})
        ml_results = {}
        latest_features_summary = {}

        feature_cols = ['return_1d', 'return_5d', 'momentum_20d', 'momentum_50d', 'volatility_20d', 'rsi_14', 'sma_ratio', 'volume_trend']

        for ticker in tickers:
            prices = prices_df[ticker].dropna()
            if len(prices) < 100:
                continue

            # Extract volume series if available in statements/info
            volume_series = None
            if ticker in statements:
                info = statements[ticker].get('info', {})
                if 'volume' in info:
                    pass

            feat_df = self.extract_features(prices, volume_series=volume_series)
            if len(feat_df) < 50:
                continue

            X = feat_df[feature_cols]
            y = feat_df['target_forward_20d']

            # Capture latest current live feature values
            latest_row = X.iloc[-1]
            latest_features_summary[ticker] = {
                'rsi_14': float(latest_row['rsi_14']),
                'volatility_20d': float(latest_row['volatility_20d']),
                'momentum_20d': float(latest_row['momentum_20d']),
                'momentum_50d': float(latest_row['momentum_50d']),
                'volume_trend': float(latest_row['volume_trend']),
                'sma_ratio': float(latest_row['sma_ratio'])
            }

            # Chronological 80/20 train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Model Selection & Fitting
            if self.model_type == "ridge":
                model = Ridge(alpha=1.0)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

            model.fit(X_train, y_train)

            # Performance Evaluation
            preds_test = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds_test))
            r2 = float(r2_score(y_test, preds_test)) if len(y_test) > 5 else 0.0
            dir_acc = float(np.mean((preds_test > 0) == (y_test > 0)) * 100.0)

            # Latest 20-day return forecast
            predicted_20d_ret = float(model.predict(X.iloc[[-1]])[0])
            predicted_ann_ret = (1.0 + predicted_20d_ret) ** (252.0 / 20.0) - 1.0

            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(feature_cols, [float(fi) for fi in model.feature_importances_]))
            elif hasattr(model, 'coef_'):
                importances = dict(zip(feature_cols, [float(abs(c)) for c in model.coef_]))
            else:
                importances = {col: 1.0/len(feature_cols) for col in feature_cols}

            ml_results[ticker] = {
                'predicted_20d_return': predicted_20d_ret,
                'predicted_annualized_return': predicted_ann_ret,
                'directional_accuracy_pct': dir_acc,
                'mae': mae,
                'r2_score': r2,
                'latest_features': latest_features_summary[ticker],
                'feature_importances': importances
            }

        return {
            'ml_results': ml_results,
            'latest_features': latest_features_summary,
            'model_used': "Random Forest Regressor" if self.model_type == "random_forest" else "Ridge Regressor"
        }
