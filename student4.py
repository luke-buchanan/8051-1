# student.py
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Student:
    """
    Stock prediction model with multiple algorithm support and enhanced features.
    """

    def __init__(
        self,
        config=None,
        random_state: int = 42,
        *,
        model_type="ensemble",  # Options: "ridge", "gbm", "rf", "nn", "lgbm", "ensemble"
        
        # Feature knobs
        n_lags=10,  # Reduced from 20 - less noise
        mom_windows=(5, 10, 20),  # Reduced windows
        vol_window=20,
        sma_windows=(10, 20, 50),
        ema_windows=(12, 26),  # Reduced
        ema_return_windows=(5, 10),  # Reduced
        rsi_window=14,
        volume_sma_windows=(10, 20),
        skew_window=20,
        kurtosis_window=20,
        vol_regime_window=60,

        # Model selection knobs
        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        cv_splits=3,
        min_train_points=20,

        # GBM knobs - IMPROVED
        gbm_n_estimators=100,  # Increased
        gbm_learning_rate=0.05,  # Increased
        gbm_max_depth=4,  # Increased
        gbm_subsample=0.7,
        gbm_min_samples_split=50,  # Added regularization

        # RF knobs - IMPROVED
        rf_n_estimators=200,  # Increased
        rf_max_depth=10,  # Increased
        rf_min_samples_split=30,
        rf_min_samples_leaf=15,  # Added

        # Neural Network knobs - IMPROVED
        nn_hidden_layers=(64, 32, 16),  # Deeper network
        nn_learning_rate=0.001,
        nn_max_iter=1000,  # More epochs
        nn_alpha=0.001,  # More regularization
        nn_early_stopping=True,

        # LightGBM knobs - IMPROVED
        lgbm_n_estimators=200,  # Increased
        lgbm_learning_rate=0.03,  # Adjusted
        lgbm_num_leaves=31,  # Increased
        lgbm_min_data_in_leaf=30,  # Increased
        lgbm_feature_fraction=0.7,  # More regularization
        lgbm_max_depth=5,  # Added depth limit

        **kwargs
    ):
        
        self.model_type = str(model_type).lower()

        # Feature parameters
        self.n_lags = int(n_lags)
        self.mom_windows = tuple(int(w) for w in mom_windows)
        self.vol_window = int(vol_window)
        self.sma_windows = tuple(int(w) for w in sma_windows)
        self.ema_windows = tuple(int(w) for w in ema_windows)
        self.ema_return_windows = tuple(int(w) for w in ema_return_windows)
        self.rsi_window = int(rsi_window)
        self.volume_sma_windows = tuple(int(w) for w in volume_sma_windows)
        self.skew_window = int(skew_window)
        self.kurtosis_window = int(kurtosis_window)
        self.vol_regime_window = int(vol_regime_window)

        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.cv_splits = int(cv_splits)

        # GBM parameters
        self.gbm_n_estimators = int(gbm_n_estimators)
        self.gbm_learning_rate = float(gbm_learning_rate)
        self.gbm_max_depth = int(gbm_max_depth)
        self.gbm_subsample = float(gbm_subsample)
        self.gbm_min_samples_split = int(gbm_min_samples_split)

        # RF parameters
        self.rf_n_estimators = int(rf_n_estimators)
        self.rf_max_depth = int(rf_max_depth)
        self.rf_min_samples_split = int(rf_min_samples_split)
        self.rf_min_samples_leaf = int(rf_min_samples_leaf)

        # Neural Network parameters
        self.nn_hidden_layers = tuple(int(x) for x in nn_hidden_layers)
        self.nn_learning_rate = float(nn_learning_rate)
        self.nn_max_iter = int(nn_max_iter)
        self.nn_alpha = float(nn_alpha)
        self.nn_early_stopping = bool(nn_early_stopping)

        # LightGBM parameters
        self.lgbm_n_estimators = int(lgbm_n_estimators)
        self.lgbm_learning_rate = float(lgbm_learning_rate)
        self.lgbm_num_leaves = int(lgbm_num_leaves)
        self.lgbm_min_data_in_leaf = int(lgbm_min_data_in_leaf)
        self.lgbm_feature_fraction = float(lgbm_feature_fraction)
        self.lgbm_max_depth = int(lgbm_max_depth)

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
        self.scaler_ = None
        self.best_alpha_ = None
        self.fitted_ = False
        self.feature_importance_ = None

    # ---------- helpers ----------

    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        return X["Close"] if "Close" in X.columns else X.iloc[:, 0]

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        series = pd.Series(series).astype(float)
        return np.log(series / series.shift(1))

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        """RSI in [0,1] using Wilder's smoothing (causal)."""
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
        """Return finite mean of y or 0.0 if none available."""
        yv = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(yv) == 0:
            return 0.0
        m = float(yv.mean())
        return m if np.isfinite(m) else 0.0

    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature engineering with interaction terms.
        """
        close = self._close_series(X).astype(float)
        lr = self._log_returns(close)

        feats = {}

        # ========== RETURN-BASED FEATURES ==========
        
        # Recent lags (most predictive)
        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)

        # Momentum
        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()

        # Volatility
        wv = self.vol_window
        feats[f"vol_{wv}"] = lr.rolling(wv, min_periods=wv).std(ddof=0)

        # EMA of returns
        for w in self.ema_return_windows:
            feats[f"ema_ret_{w}"] = lr.ewm(span=w, adjust=False, min_periods=w).mean()

        # Skewness and Kurtosis
        feats[f"skew_{self.skew_window}"] = lr.rolling(
            self.skew_window, min_periods=self.skew_window
        ).skew()
        feats[f"kurt_{self.kurtosis_window}"] = lr.rolling(
            self.kurtosis_window, min_periods=self.kurtosis_window
        ).kurt()

        # ========== PRICE-BASED FEATURES ==========

        # SMA distance
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f"price_to_sma_{w}"] = (close - sma) / sma.replace(0, np.nan)

        # EMA distance
        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            feats[f"ema_dist_{w}"] = (close - ema) / ema.replace(0, np.nan)

        # RSI
        feats[f"rsi_{self.rsi_window}"] = self._rsi(close, self.rsi_window)
        # RSI deviation from neutral
        feats[f"rsi_dev"] = self._rsi(close, self.rsi_window) - 0.5

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        feats['macd_hist'] = macd_line - macd_signal
        # Normalized MACD
        feats['macd_norm'] = (macd_line - macd_signal) / close.replace(0, np.nan)

        # ========== INTERACTION FEATURES ==========
        
        # Momentum * Volatility (trending + volatility)
        feats['mom_vol_5'] = feats['mom_5'] * feats[f'vol_{wv}']
        feats['mom_vol_10'] = feats['mom_10'] * feats[f'vol_{wv}']

        # ========== HIGH/LOW/OPEN FEATURES ==========
        
        if "High" in X.columns and "Low" in X.columns:
            high = X["High"].astype(float)
            low = X["Low"].astype(float)
            feats["hl_range"] = (high - low) / close.replace(0, np.nan)
            
        if "Open" in X.columns:
            open_price = X["Open"].astype(float)
            feats["oc_ratio"] = (close - open_price) / open_price.replace(0, np.nan)

        # ========== VOLUME FEATURES ==========
        
        if "Volume" in X.columns:
            volume = X["Volume"].astype(float)
            for w in self.volume_sma_windows:
                vol_sma = volume.rolling(w, min_periods=w).mean()
                feats[f"vol_ratio_{w}"] = volume / vol_sma.replace(0, np.nan)

        # ========== TIME FEATURES ==========
        
        feats["month"] = X.index.month
        feats["quarter"] = X.index.quarter
        feats["day_of_week"] = X.index.dayofweek

        # ========== VOLATILITY REGIME ==========
        
        vol_long = lr.rolling(self.vol_regime_window, min_periods=self.vol_regime_window).std(ddof=0)
        vol_current = lr.rolling(self.vol_window, min_periods=self.vol_window).std(ddof=0)
        feats["vol_regime"] = vol_current / vol_long.replace(0, np.nan)

        # Create DataFrame and clean
        F = pd.DataFrame(feats, index=X.index).replace([np.inf, -np.inf], np.nan)
        F = F.dropna()
        return F

    # ---------- fit / predict ----------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        """
        Train model with improved hyperparameters and regularization.
        """
        F = self._make_features(X_train)

        # Fallback if we can't compute features yet
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

        if len(F) < self.min_train_points:
            mean_y = float(y.mean()) if np.isfinite(y.mean()) else 0.0
            self.pipe_ = DummyRegressor(strategy="constant", constant=mean_y)
            self.pipe_.fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

        # ===== MODEL SELECTION =====
        
        if self.model_type == "gbm":
            self.pipe_ = GradientBoostingRegressor(
                n_estimators=self.gbm_n_estimators,
                learning_rate=self.gbm_learning_rate,
                max_depth=self.gbm_max_depth,
                subsample=self.gbm_subsample,
                min_samples_split=self.gbm_min_samples_split,
                random_state=self.random_state,
                loss='huber',
                alpha=0.9
            )
            self.pipe_.fit(F.values, y.values)
            self.feature_importance_ = self.pipe_.feature_importances_
            
        elif self.model_type == "rf":
            self.pipe_ = RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.pipe_.fit(F.values, y.values)
            self.feature_importance_ = self.pipe_.feature_importances_

        elif self.model_type == "lgbm":
            self.pipe_ = lgb.LGBMRegressor(
                n_estimators=self.lgbm_n_estimators,
                learning_rate=self.lgbm_learning_rate,
                num_leaves=self.lgbm_num_leaves,
                max_depth=self.lgbm_max_depth,
                min_data_in_leaf=self.lgbm_min_data_in_leaf,
                feature_fraction=self.lgbm_feature_fraction,
                bagging_fraction=0.7,
                bagging_freq=5,
                lambda_l1=0.1,  # L1 regularization
                lambda_l2=0.1,  # L2 regularization
                objective='regression',
                metric='rmse',
                random_state=self.random_state,
                verbose=-1
            )
            self.pipe_.fit(F.values, y.values)
            self.feature_importance_ = self.pipe_.feature_importances_
            
        elif self.model_type == "nn":
            self.scaler_ = StandardScaler()
            F_scaled = self.scaler_.fit_transform(F.values)
            
            self.pipe_ = MLPRegressor(
                hidden_layer_sizes=self.nn_hidden_layers,
                activation='relu',
                solver='adam',
                learning_rate_init=self.nn_learning_rate,
                max_iter=self.nn_max_iter,
                alpha=self.nn_alpha,
                early_stopping=self.nn_early_stopping,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=self.random_state,
                verbose=False
            )
            self.pipe_.fit(F_scaled, y.values)

        elif self.model_type == "ensemble":
            # Ensemble of multiple models
            lgbm_model = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=5,
                random_state=self.random_state,
                verbose=-1
            )
            
            gbm_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.7,
                random_state=self.random_state
            )
            
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=30,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.pipe_ = VotingRegressor([
                ('lgbm', lgbm_model),
                ('gbm', gbm_model),
                ('rf', rf_model)
            ])
            self.pipe_.fit(F.values, y.values)
            
        else:  # Ridge with CV
            n_splits = min(self.cv_splits, max(2, len(F) // 200))
            tscv = TimeSeriesSplit(n_splits=n_splits)

            best_alpha, best_mse = None, np.inf
            for a in self.alpha_grid:
                mses = []
                for tr_idx, va_idx in tscv.split(F.values):
                    X_tr, X_va = F.values[tr_idx], F.values[va_idx]
                    y_tr, y_va = y.values[tr_idx], y.values[va_idx]
                    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a))])
                    pipe.fit(X_tr, y_tr)
                    y_hat = pipe.predict(X_va)
                    mses.append(mean_squared_error(y_va, y_hat))
                avg_mse = float(np.mean(mses)) if mses else np.inf
                if avg_mse < best_mse:
                    best_mse, best_alpha = avg_mse, a

            if best_alpha is None:
                best_alpha = 1.0
            self.best_alpha_ = float(best_alpha)

            self.pipe_ = Pipeline([
                ("scaler", StandardScaler()), 
                ("model", Ridge(alpha=self.best_alpha_, random_state=self.random_state))
            ])
            self.pipe_.fit(F.values, y.values)

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:
        """
        Return predictions of next-H-day cumulative log returns.
        """
        F = self._make_features(X)
        if not self.fitted_ or self.pipe_ is None:
            idx = F.index if len(F) else X.index
            return pd.Series(0.0, index=idx, name="y_pred")
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")
        
        # Neural networks need scaled features
        if self.model_type == "nn" and self.scaler_ is not None:
            F_scaled = self.scaler_.transform(F.values)
            y_hat = self.pipe_.predict(F_scaled)
        else:
            y_hat = self.pipe_.predict(F.values)
        
        return pd.Series(y_hat, index=F.index, name="y_pred")