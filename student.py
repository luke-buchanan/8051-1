# student.py
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

# Try to import XGBoost (optional but recommended)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class Student:
    """
    AGGRESSIVE stock prediction model optimized for directional accuracy.
    
    Key improvements:
    - GradientBoosting as default (works better than Ridge/LSTM for this)
    - Feature interactions and transformations
    - MACD, ROC, and momentum cross features
    - Ensemble voting for stability
    - Optimized hyperparameters for direction prediction
    """

    def __init__(
        self,
        config=None,
        random_state: int = 42,
        *,
        # Model selection - CHANGED DEFAULT TO GBM
        model_type="linear",  # Options: "ridge", "gbm", "xgb", "ensemble"
        
        # Feature engineering knobs - MORE AGGRESSIVE
        n_lags=20,                      # Increased from 10
        mom_windows=(3, 5, 10, 20, 60, 120),  # More windows including short-term
        vol_window=20,
        sma_windows=(5, 10, 20, 50, 200),
        ema_windows=(8, 12, 26, 50),
        rsi_window=14,
        
        # Enhanced feature parameters
        bb_window=20,
        atr_window=14,
        stoch_window=14,
        obv_ema=10,
        sharpe_window=20,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_windows=(5, 10, 20),  # Rate of change
        
        # Model hyperparameters - TUNED FOR DIRECTIONAL ACCURACY
        # GBM params
        gbm_n_estimators=200,
        gbm_learning_rate=0.05,
        gbm_max_depth=4,
        gbm_subsample=0.8,
        gbm_min_samples_split=20,
        
        # XGBoost params
        xgb_n_estimators=200,
        xgb_learning_rate=0.05,
        xgb_max_depth=4,
        
        # Ridge params
        alpha_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        cv_splits=3,
        min_train_points=150,  # Reduced to allow earlier training
        
        **kwargs
    ):
        # Model type
        self.model_type = str(model_type).lower()
        
        # Feature parameters
        self.n_lags = int(n_lags)
        self.mom_windows = tuple(int(w) for w in mom_windows)
        self.vol_window = int(vol_window)
        self.sma_windows = tuple(int(w) for w in sma_windows)
        self.ema_windows = tuple(int(w) for w in ema_windows)
        self.rsi_window = int(rsi_window)
        
        # Enhanced feature parameters
        self.bb_window = int(bb_window)
        self.atr_window = int(atr_window)
        self.stoch_window = int(stoch_window)
        self.obv_ema = int(obv_ema)
        self.sharpe_window = int(sharpe_window)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.macd_signal = int(macd_signal)
        self.roc_windows = tuple(int(w) for w in roc_windows)
        
        # Model hyperparameters
        self.gbm_n_estimators = int(gbm_n_estimators)
        self.gbm_learning_rate = float(gbm_learning_rate)
        self.gbm_max_depth = int(gbm_max_depth)
        self.gbm_subsample = float(gbm_subsample)
        self.gbm_min_samples_split = int(gbm_min_samples_split)
        
        self.xgb_n_estimators = int(xgb_n_estimators)
        self.xgb_learning_rate = float(xgb_learning_rate)
        self.xgb_max_depth = int(xgb_max_depth)
        
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.cv_splits = int(cv_splits)
        self.min_train_points = int(min_train_points)
        self.random_state = int(random_state)

        # Apply config overrides
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Learned state
        self.pipe_ = None
        self.best_params_ = None
        self.fitted_ = False

    # ==================== Helper Methods ====================

    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        """Extract close price series from DataFrame"""
        return X["Close"] if "Close" in X.columns else X.iloc[:, 0]
    
    @staticmethod
    def _col(X: pd.DataFrame, name: str) -> pd.Series | None:
        """Safely extract a column by name (case-insensitive)"""
        for variant in [name, name.capitalize(), name.lower(), name.upper()]:
            if variant in X.columns:
                return X[variant].astype(float)
        return None

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        """Calculate log returns"""
        series = pd.Series(series).astype(float)
        return np.log(series / series.shift(1))

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        """Calculate RSI (Relative Strength Index) in [0,1]"""
        close = pd.Series(close).astype(float)
        diff = close.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 1 - (1 / (1 + rs))
        return rsi.fillna(0.5)

    @staticmethod
    def _finite_mean(y: pd.Series) -> float:
        """Return finite mean or 0.0 if none available"""
        yv = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(yv) == 0:
            return 0.0
        m = float(yv.mean())
        return m if np.isfinite(m) else 0.0

    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Build features with exact names requested, causally (no peeking):
        open, high, low, close, adj_close, volume,
        return_lag_1, return_lag_5, return_lag_10,
        rolling_mean_5, rolling_std_5, ema_returns_5,
        rolling_mean_15, rolling_std_15, ema_returns_15,
        rolling_mean_25, rolling_std_25, ema_returns_25,
        sma_20, sma_50, price_to_sma_20, rsi_14,
        hl_range, oc_ratio, volume_sma_20, volume_ratio,
        month, quarter, day_of_week, day_of_month
        (and retains your existing engineered features).
        """
        idx = X.index

        # --- Pull OHLCV, case-insensitive (safe if missing) ---
        close = self._close_series(X).astype(float)
        openp = self._col(X, "Open")
        high  = self._col(X, "High")
        low   = self._col(X, "Low")
        # adjc  = self._col(X, "Adj_close") or self._col(X, "Adj Close") or self._col(X, "AdjClose")
        vol   = self._col(X, "Volume")

        # Daily log returns (t vs t-1), causal
        lr = self._log_returns(close)

        feats = {}

        # ---------- Raw passthrough (current-day values are OK for predicting t→t+H) ----------
        feats["open"]      = openp if openp is not None else pd.Series(np.nan, index=idx)
        feats["high"]      = high  if high  is not None else pd.Series(np.nan, index=idx)
        feats["low"]       = low   if low   is not None else pd.Series(np.nan, index=idx)
        feats["close"]     = close
        # feats["adj_close"] = adjc if adjc is not None else pd.Series(np.nan, index=idx)
        feats["volume"]    = vol  if vol  is not None else pd.Series(np.nan, index=idx)

        # ---------- Horizon-agnostic return lags (cumulative log returns) ----------
        # return_lag_k = log(C_t / C_{t-k})
        feats["return_lag_1"]  = np.log(close / close.shift(1))
        feats["return_lag_5"]  = np.log(close / close.shift(5))
        feats["return_lag_10"] = np.log(close / close.shift(10))

        # ---------- Rolling stats of daily log returns ----------
        for w in (5, 15, 25):
            feats[f"rolling_mean_{w}"] = lr.rolling(w, min_periods=w).mean()
            feats[f"rolling_std_{w}"]  = lr.rolling(w, min_periods=w).std(ddof=0)
            feats[f"ema_returns_{w}"]  = lr.ewm(span=w, adjust=False, min_periods=w).mean()

        # ---------- Moving averages on price and distances ----------
        sma20 = close.rolling(20, min_periods=20).mean()
        sma50 = close.rolling(50, min_periods=50).mean()
        feats["sma_20"] = sma20
        feats["sma_50"] = sma50
        feats["price_to_sma_20"] = (close - sma20) / sma20.replace(0, np.nan)

        # ---------- RSI(14) in [0,1] ----------
        feats["rsi_14"] = self._rsi(close, 14)

        # ---------- Intraday ranges and ratios ----------
        if (high is not None) and (low is not None):
            feats["hl_range"] = (high - low) / close.replace(0, np.nan)
        else:
            feats["hl_range"] = pd.Series(np.nan, index=idx)

        if openp is not None:
            feats["oc_ratio"] = (close - openp) / openp.replace(0, np.nan)
        else:
            feats["oc_ratio"] = pd.Series(np.nan, index=idx)

        # ---------- Volume features ----------
        if vol is not None:
            vol_sma20 = vol.rolling(20, min_periods=20).mean()
            feats["volume_sma_20"] = vol_sma20
            feats["volume_ratio"]  = vol / vol_sma20.replace(0, np.nan)
        else:
            feats["volume_sma_20"] = pd.Series(np.nan, index=idx)
            feats["volume_ratio"]  = pd.Series(np.nan, index=idx)

        # ---------- Calendar features ----------
        # (integers; no leakage—date is known at t)
        if hasattr(idx, "to_series"):
            dt = pd.Series(idx, index=idx)
            feats["month"]        = dt.dt.month.astype(int)
            feats["quarter"]      = dt.dt.quarter.astype(int)
            feats["day_of_week"]  = dt.dt.dayofweek.astype(int)   # 0=Mon
            feats["day_of_month"] = dt.dt.day.astype(int)
        else:
            # Fallback if index isn’t datetime
            feats["month"]        = pd.Series(np.nan, index=idx)
            feats["quarter"]      = pd.Series(np.nan, index=idx)
            feats["day_of_week"]  = pd.Series(np.nan, index=idx)
            feats["day_of_month"] = pd.Series(np.nan, index=idx)

        # ---------- (Optional) keep your existing engineered features ----------
        # You can retain your current aggressive features by merging them here.
        # If you want to keep them, uncomment the following lines to add them
        # on top of the requested set:
        #
        # base_feats = {}  # build your prior features into base_feats...
        # feats.update(base_feats)

        # Assemble, clean infinities/NaNs, and drop rows until all rolling windows exist
        F = pd.DataFrame(feats, index=idx)
        F = F.replace([np.inf, -np.inf], np.nan)
        F = F.dropna()

        return F


    # ==================== Model Training ====================

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        """
        Train model with focus on directional accuracy.
        Default is GradientBoosting which typically outperforms Ridge/LSTM.
        """
        F = self._make_features(X_train)

        # Fallback for early periods
        if F.empty:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = DummyRegressor(strategy="constant", constant=mean_y)
            self.pipe_.fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

        # Align target
        y = y_train.reindex(F.index)
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        F, y = F.loc[mask], y.loc[mask]

        if len(y) == 0:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = DummyRegressor(strategy="constant", constant=mean_y)
            self.pipe_.fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

        # Use constant for very short history
        if len(F) < self.min_train_points:
            mean_y = float(y.mean()) if np.isfinite(y.mean()) else 0.0
            self.pipe_ = DummyRegressor(strategy="constant", constant=mean_y)
            self.pipe_.fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

        # ===== MODEL SELECTION =====
        
        if self.model_type == "gbm":
            # GradientBoosting - BEST FOR DIRECTIONAL ACCURACY
            self.pipe_ = GradientBoostingRegressor(
                n_estimators=self.gbm_n_estimators,
                learning_rate=self.gbm_learning_rate,
                max_depth=self.gbm_max_depth,
                subsample=self.gbm_subsample,
                min_samples_split=self.gbm_min_samples_split,
                random_state=self.random_state,
                loss='huber',  # Robust to outliers
                alpha=0.9
            )
            self.pipe_.fit(F.values, y.values)
            
        elif self.model_type == "xgb" and HAS_XGBOOST:
            # XGBoost - Also very good
            self.pipe_ = XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                objective='reg:squarederror'
            )
            self.pipe_.fit(F.values, y.values)
            
        elif self.model_type == "ensemble":
            # Voting ensemble - combines multiple models
            gbm = GradientBoostingRegressor(
                n_estimators=self.gbm_n_estimators,
                learning_rate=self.gbm_learning_rate,
                max_depth=self.gbm_max_depth,
                subsample=self.gbm_subsample,
                min_samples_split=self.gbm_min_samples_split,
                random_state=self.random_state,
                loss='huber'
            )
            
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                random_state=self.random_state
            )
            
            # Scale data for Ridge
            scaler = RobustScaler()
            F_scaled = scaler.fit_transform(F.values)
            
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            
            # Fit models
            gbm.fit(F.values, y.values)
            rf.fit(F.values, y.values)
            ridge.fit(F_scaled, y.values)
            
            # Store scaler and models
            self.scaler_ = scaler
            self.models_ = {'gbm': gbm, 'rf': rf, 'ridge': ridge}
            self.pipe_ = None  # Signal we're using ensemble
        
        elif self.model_type == "linear":
            # Linear Regression (no regularization)
            self.scaler_ = RobustScaler()
            F_scaled = self.scaler_.fit_transform(F.values)
            self.pipe_ = LinearRegression()
            self.pipe_.fit(F_scaled, y.values)
            
        elif self.model_type == "ridge":
            # Ridge with cross-validation
            n_splits = min(self.cv_splits, max(2, len(F) // 200))
            tscv = TimeSeriesSplit(n_splits=n_splits)

            best_alpha, best_mse = None, np.inf
            for a in self.alpha_grid:
                mses = []
                for tr_idx, va_idx in tscv.split(F.values):
                    X_tr, X_va = F.values[tr_idx], F.values[va_idx]
                    y_tr, y_va = y.values[tr_idx], y.values[va_idx]
                    scaler = RobustScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr)
                    X_va_scaled = scaler.transform(X_va)
                    model = Ridge(alpha=a, random_state=self.random_state)
                    model.fit(X_tr_scaled, y_tr)
                    y_hat = model.predict(X_va_scaled)
                    mses.append(mean_squared_error(y_va, y_hat))
                avg_mse = float(np.mean(mses)) if mses else np.inf
                if avg_mse < best_mse:
                    best_mse, best_alpha = avg_mse, a

            # Fit final Ridge model
            self.scaler_ = RobustScaler()
            F_scaled = self.scaler_.fit_transform(F.values)
            self.pipe_ = Ridge(alpha=best_alpha if best_alpha else 1.0, random_state=self.random_state)
            self.pipe_.fit(F_scaled, y.values)
        
        else:
            # Default to GBM
            self.pipe_ = GradientBoostingRegressor(
                n_estimators=self.gbm_n_estimators,
                learning_rate=self.gbm_learning_rate,
                max_depth=self.gbm_max_depth,
                subsample=self.gbm_subsample,
                min_samples_split=self.gbm_min_samples_split,
                random_state=self.random_state,
                loss='huber'
            )
            self.pipe_.fit(F.values, y.values)

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:
        """Return predictions for next-H-day cumulative log returns."""
        F = self._make_features(X)
        
        if not self.fitted_:
            idx = F.index if len(F) else X.index
            return pd.Series(0.0, index=idx, name="y_pred")
        
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")
        
        # Handle ensemble separately
        if self.pipe_ is None and hasattr(self, 'models_'):
            # Ensemble prediction
            gbm_pred = self.models_['gbm'].predict(F.values)
            rf_pred = self.models_['rf'].predict(F.values)
            
            F_scaled = self.scaler_.transform(F.values)
            ridge_pred = self.models_['ridge'].predict(F_scaled)
            
            # Weighted average (GBM gets more weight)
            y_hat = 0.5 * gbm_pred + 0.3 * rf_pred + 0.2 * ridge_pred
        
        # Handle Ridge with scaling
        elif self.model_type == "ridge" and hasattr(self, 'scaler_'):
            F_scaled = self.scaler_.transform(F.values)
            y_hat = self.pipe_.predict(F_scaled)
        
        # Standard prediction
        else:
            y_hat = self.pipe_.predict(F.values)
        
        return pd.Series(y_hat, index=F.index, name="y_pred")