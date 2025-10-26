# student.py - Super Ensemble
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


class Student:
    """
    Super Ensemble: GBM + Ridge + RandomForest
    """

    def __init__(
        self,
        config=None,
        random_state: int = 42,
        **kwargs
    ):
        self.random_state = int(random_state)
        
        # Feature parameters
        self.n_lags = 20
        self.mom_windows = (5, 10, 20, 60)
        self.vol_window = 20
        self.sma_windows = (10, 20, 50)
        self.rsi_window = 14
        self.min_train_points = 30
        
        # Model parameters - aggressive
        self.gbm_n_estimators = 60
        self.gbm_learning_rate = 0.03
        self.gbm_max_depth = 4
        
        self.rf_n_estimators = 100
        self.rf_max_depth = 10
        
        # Apply config overrides
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Learned state
        self.models_ = {}
        self.scaler_ = None
        self.fitted_ = False

    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        return X["Close"] if "Close" in X.columns else X.iloc[:, 0]

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        series = pd.Series(series).astype(float)
        return np.log(series / series.shift(1))

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
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
        yv = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(yv) == 0:
            return 0.0
        m = float(yv.mean())
        return m if np.isfinite(m) else 0.0

    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:
        close = self._close_series(X).astype(float)
        lr = self._log_returns(close)
        feats = {}

        # Lags (20)
        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)

        # Momentum (4)
        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()

        # Volatility (1)
        feats[f"vol_{self.vol_window}"] = lr.rolling(self.vol_window, min_periods=self.vol_window).std(ddof=0)

        # SMA distances (3)
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f"sma_dist_{w}"] = (close - sma) / sma.replace(0, np.nan)

        # RSI (1)
        feats[f"rsi_{self.rsi_window}"] = self._rsi(close, self.rsi_window)

        # MACD (3)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        feats['macd_line'] = ema_12 - ema_26
        feats['macd_signal'] = feats['macd_line'].ewm(span=9, adjust=False).mean()
        feats['macd_hist'] = feats['macd_line'] - feats['macd_signal']
        
        # Additional safe features for ensemble
        # Short-term volatility (raw, not ratio)
        feats['vol_5'] = lr.rolling(5, min_periods=5).std(ddof=0)
        
        # Volatility acceleration (change in volatility)
        vol_10 = lr.rolling(10, min_periods=10).std(ddof=0)
        vol_30 = lr.rolling(30, min_periods=30).std(ddof=0)
        feats['vol_change'] = vol_10 - vol_30

        F = pd.DataFrame(feats, index=X.index).replace([np.inf, -np.inf], np.nan)
        F = F.dropna()
        return F

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        F = self._make_features(X_train)

        if F.empty:
            mean_y = self._finite_mean(y_train)
            self.models_['dummy'] = DummyRegressor(strategy="constant", constant=mean_y)
            self.models_['dummy'].fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

        y = y_train.reindex(F.index)
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        F, y = F.loc[mask], y.loc[mask]

        if len(y) == 0 or len(F) < self.min_train_points:
            mean_y = self._finite_mean(y_train) if len(y) == 0 else float(y.mean())
            self.models_['dummy'] = DummyRegressor(strategy="constant", constant=mean_y)
            self.models_['dummy'].fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

        # Train GBM (50% weight)
        self.models_['gbm'] = GradientBoostingRegressor(
            n_estimators=self.gbm_n_estimators,
            learning_rate=self.gbm_learning_rate,
            max_depth=self.gbm_max_depth,
            subsample=0.8,
            random_state=self.random_state,
            loss='huber',
            alpha=0.9
        )
        self.models_['gbm'].fit(F.values, y.values)

        # Train RandomForest (30% weight)
        self.models_['rf'] = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models_['rf'].fit(F.values, y.values)

        # Train Ridge (20% weight)
        self.scaler_ = StandardScaler()
        F_scaled = self.scaler_.fit_transform(F.values)
        self.models_['ridge'] = Ridge(alpha=1.0, random_state=self.random_state)
        self.models_['ridge'].fit(F_scaled, y.values)

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:
        F = self._make_features(X)
        if not self.fitted_:
            idx = F.index if len(F) else X.index
            return pd.Series(0.0, index=idx, name="y_pred")
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")

        # Dummy fallback
        if 'dummy' in self.models_:
            y_hat = self.models_['dummy'].predict(F.values if len(F.values) > 0 else [[0.0]])
            if len(y_hat) == 1 and len(F) > 1:
                y_hat = np.full(len(F), y_hat[0])
            return pd.Series(y_hat, index=F.index, name="y_pred")

        # Super ensemble prediction
        gbm_pred = self.models_['gbm'].predict(F.values)
        rf_pred = self.models_['rf'].predict(F.values)
        
        F_scaled = self.scaler_.transform(F.values)
        ridge_pred = self.models_['ridge'].predict(F_scaled)
        
        # Weighted average: GBM 50%, RF 30%, Ridge 20%
        y_hat = 0.5 * gbm_pred + 0.3 * rf_pred + 0.2 * ridge_pred

        return pd.Series(y_hat, index=F.index, name="y_pred")