# student.py
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Student:
    def __init__(
        self,
        config=None,
        random_state: int = 42,
        *,
        model_type="gbm",
        
        n_lags=20,
        mom_windows=(5, 10, 20, 60),
        vol_window=20,
        sma_windows=(10, 20, 50),
        ema_windows=(8, 12, 26),
        rsi_window=14,

        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        cv_splits=3,
        min_train_points=20,

        gbm_n_estimators=40,
        gbm_learning_rate=0.02,
        gbm_max_depth=3,
        gbm_subsample=0.8,

        **kwargs
    ):
        
        self.model_type = str(model_type).lower()

        self.n_lags = int(n_lags)
        self.mom_windows = tuple(int(w) for w in mom_windows)
        self.vol_window = int(vol_window)
        self.sma_windows = tuple(int(w) for w in sma_windows)
        self.ema_windows = tuple(int(w) for w in ema_windows)
        self.rsi_window = int(rsi_window)

        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.cv_splits = int(cv_splits)

        self.gbm_n_estimators = int(gbm_n_estimators)
        self.gbm_learning_rate = float(gbm_learning_rate)
        self.gbm_max_depth = int(gbm_max_depth)
        self.gbm_subsample = float(gbm_subsample)

        self.min_train_points = int(min_train_points)
        self.random_state = int(random_state)

        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.pipe_ = None
        self.scaler_ = None
        self.best_alpha_ = None
        self.fitted_ = False

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
        x_copy = X.copy()

        feats = {}

        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)

        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()

        wv = self.vol_window
        feats[f"vol_{wv}"] = lr.rolling(wv, min_periods=wv).std(ddof=0)

        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
            feats[f"ema_dist_{w}"] = (close - ema) / ema.replace(0, np.nan)

        feats[f"rsi_{self.rsi_window}"] = self._rsi(close, self.rsi_window)

        for w in self.sma_windows:
            feats[f"roll_mean_{w}"] = lr.rolling(w, min_periods=w).mean()
            feats[f"roll_std_{w}"] = lr.rolling(w, min_periods=w).std(ddof=0)

        for w in [20, 50]:
            sma = close.shift(1).rolling(w).mean()
            feats[f"sma_dist_{w}"] = (close.shift(1) - sma) / sma.replace(0, np.nan)

        if isinstance(x_copy.index, pd.DatetimeIndex):
            feats['qtr'] = x_copy.index.quarter
            feats['mo'] = x_copy.index.month

        F = pd.DataFrame(feats, index=X.index).replace([np.inf, -np.inf], np.nan)
        F = F.dropna()
        return F

    # ---------- fit / predict ----------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        F = self._make_features(X_train)

        if F.empty:
            mean_y = self._finite_mean(y_train)
            self.pipe_ = DummyRegressor(strategy="constant", constant=mean_y)
            self.pipe_.fit([[0.0]], [0.0])
            self.fitted_ = True
            return self

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
        
        self.pipe_ = GradientBoostingRegressor(
            n_estimators=self.gbm_n_estimators,
            learning_rate=self.gbm_learning_rate,
            max_depth=self.gbm_max_depth,
            subsample=self.gbm_subsample,
            random_state=self.random_state,
            loss='huber',
            alpha=0.9
        )

        self.pipe_.fit(F.values, y.values)

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:

        F = self._make_features(X)
        if not self.fitted_ or self.pipe_ is None:
            idx = F.index if len(F) else X.index
            return pd.Series(0.0, index=idx, name="y_pred")
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")
        
        y_hat = self.pipe_.predict(F.values)
        
        return pd.Series(y_hat, index=F.index, name="y_pred")