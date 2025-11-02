# student.py
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Student:
    """
    Stock prediction model with Optuna hyperparameter tuning.
    Uses file-based caching to persist parameters across runs.
    """
    
    # Class-level cache
    _optuna_cache = {}
    _cache_dir = Path(".optuna_cache")

    def __init__(
        self,
        config=None,
        random_state: int = 42,
        *,
        model_type="gbm",
        
        # Optuna configuration
        use_optuna=True,
        optuna_n_trials=30,
        optuna_cv_splits=3,
        optuna_timeout=180,
        optuna_optimize_once=True,  # Only optimize on first fit
        
        # Feature knobs
        n_lags=20,
        mom_windows=(5, 10, 20, 60),
        vol_window=20,
        sma_windows=(10, 20, 50),
        ema_windows=(8, 12, 26),
        rsi_window=14,

        # Model selection knobs
        alpha_grid=(0.01, 0.1, 1.0, 10.0),
        cv_splits=3,
        min_train_points=20,

        # GBM knobs
        gbm_n_estimators=40,
        gbm_learning_rate=0.02,
        gbm_max_depth=3,
        gbm_subsample=0.8,

        # RF knobs
        rf_n_estimators=100,
        rf_max_depth=8,
        rf_min_samples_split=20,

        # Neural Network knobs
        nn_hidden_layers=(32, 16),
        nn_learning_rate=0.005,
        nn_max_iter=500,
        nn_alpha=0.0001,
        nn_early_stopping=True,

        # LightGBM knobs
        lgbm_n_estimators=100,
        lgbm_learning_rate=0.05,
        lgbm_num_leaves=16,
        lgbm_min_data_in_leaf=20,
        lgbm_feature_fraction=0.8,

        **kwargs
    ):
        
        self.model_type = str(model_type).lower()
        
        # Optuna settings
        self.use_optuna = bool(use_optuna)
        self.optuna_n_trials = int(optuna_n_trials)
        self.optuna_cv_splits = int(optuna_cv_splits)
        self.optuna_timeout = int(optuna_timeout)
        self.optuna_optimize_once = bool(optuna_optimize_once)

        # Feature parameters
        self.n_lags = int(n_lags)
        self.mom_windows = tuple(int(w) for w in mom_windows)
        self.vol_window = int(vol_window)
        self.sma_windows = tuple(int(w) for w in sma_windows)
        self.ema_windows = tuple(int(w) for w in ema_windows)
        self.rsi_window = int(rsi_window)

        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.cv_splits = int(cv_splits)

        # GBM parameters
        self.gbm_n_estimators = int(gbm_n_estimators)
        self.gbm_learning_rate = float(gbm_learning_rate)
        self.gbm_max_depth = int(gbm_max_depth)
        self.gbm_subsample = float(gbm_subsample)

        # RF parameters
        self.rf_n_estimators = int(rf_n_estimators)
        self.rf_max_depth = int(rf_max_depth)
        self.rf_min_samples_split = int(rf_min_samples_split)

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
        self.best_params_ = None
        self.fitted_ = False
        
        # Create cache directory
        if self.use_optuna and self.optuna_optimize_once:
            Student._cache_dir.mkdir(exist_ok=True)

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
        Leakage-safe features with MACD added.
        """
        close = self._close_series(X).astype(float)
        lr = self._log_returns(close)

        feats = {}

        # Lags
        for i in range(1, self.n_lags + 1):
            feats[f"lag{i}"] = lr.shift(i)

        # Momentum
        for w in self.mom_windows:
            feats[f"mom_{w}"] = lr.rolling(w, min_periods=w).mean()

        # Volatility
        wv = self.vol_window
        feats[f"vol_{wv}"] = lr.rolling(wv, min_periods=wv).std(ddof=0)

        # SMA/EMA distances
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f"sma_dist_{w}"] = (close - sma) / sma.replace(0, np.nan)

        # RSI
        feats[f"rsi_{self.rsi_window}"] = self._rsi(close, self.rsi_window)

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        feats['macd_line'] = ema_12 - ema_26
        feats['macd_signal'] = feats['macd_line'].ewm(span=9, adjust=False).mean()
        feats['macd_hist'] = feats['macd_line'] - feats['macd_signal']

        F = pd.DataFrame(feats, index=X.index).replace([np.inf, -np.inf], np.nan)
        F = F.dropna()
        return F

    # ---------- Optuna optimization with file caching ----------

    def _load_cached_params(self, cache_key: str):
        """Load parameters from file cache."""
        # Check memory cache first
        if cache_key in Student._optuna_cache:
            return Student._optuna_cache[cache_key]
        
        # Check file cache
        cache_file = Student._cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    params = pickle.load(f)
                Student._optuna_cache[cache_key] = params  # Load into memory
                return params
            except:
                pass
        
        return None

    def _save_cached_params(self, cache_key: str, params: dict):
        """Save parameters to file cache."""
        Student._optuna_cache[cache_key] = params
        
        cache_file = Student._cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(params, f)
        except:
            pass

    def _optimize_gbm_params(self, F: np.ndarray, y: np.ndarray) -> dict:
        """
        Use Optuna to find optimal GBM hyperparameters.
        """
        cache_key = f"gbm_trials{self.optuna_n_trials}_cv{self.optuna_cv_splits}"
        
        # Try to load cached params
        if self.optuna_optimize_once:
            cached = self._load_cached_params(cache_key)
            if cached is not None:
                print(f"✓ Using cached GBM parameters")
                return cached
        
        print(f"Optimizing GBM hyperparameters ({self.optuna_n_trials} trials, {self.optuna_timeout}s timeout)...")
        tscv = TimeSeriesSplit(n_splits=self.optuna_cv_splits)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'min_samples_split': trial.suggest_int('min_samples_split', 20, 100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
                'loss': 'huber',
                'alpha': 0.9,
                'random_state': self.random_state
            }
            
            scores = []
            for train_idx, val_idx in tscv.split(F):
                X_train, X_val = F[train_idx], F[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = GradientBoostingRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=self.optuna_n_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=False
        )
        
        best_params = study.best_params
        print(f"✓ Optimization complete. Best params: {best_params}")
        
        # Cache the result
        if self.optuna_optimize_once:
            self._save_cached_params(cache_key, best_params)
        
        return best_params

    def _optimize_lgbm_params(self, F: np.ndarray, y: np.ndarray) -> dict:
        """Use Optuna to find optimal LightGBM hyperparameters."""
        cache_key = f"lgbm_trials{self.optuna_n_trials}_cv{self.optuna_cv_splits}"
        
        if self.optuna_optimize_once:
            cached = self._load_cached_params(cache_key)
            if cached is not None:
                print(f"✓ Using cached LGBM parameters")
                return cached
        
        print(f"Optimizing LGBM hyperparameters ({self.optuna_n_trials} trials, {self.optuna_timeout}s timeout)...")
        tscv = TimeSeriesSplit(n_splits=self.optuna_cv_splits)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
                'objective': 'regression',
                'random_state': self.random_state,
                'verbose': -1
            }
            
            scores = []
            for train_idx, val_idx in tscv.split(F):
                X_train, X_val = F[train_idx], F[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=self.optuna_n_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=False
        )
        
        best_params = study.best_params
        print(f"✓ Optimization complete. Best params: {best_params}")
        
        if self.optuna_optimize_once:
            self._save_cached_params(cache_key, best_params)
        
        return best_params

    # ---------- fit / predict ----------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        """
        Train model with Optuna tuning (uses cached params after first optimization).
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

        F_np = F.values
        y_np = y.values

        # ===== MODEL SELECTION WITH OPTUNA =====
        
        if self.model_type == "gbm":
            if self.use_optuna:
                best_params = self._optimize_gbm_params(F_np, y_np)
                self.best_params_ = best_params
                self.pipe_ = GradientBoostingRegressor(**best_params)
            else:
                self.pipe_ = GradientBoostingRegressor(
                    n_estimators=self.gbm_n_estimators,
                    learning_rate=self.gbm_learning_rate,
                    max_depth=self.gbm_max_depth,
                    subsample=self.gbm_subsample,
                    random_state=self.random_state,
                    loss='huber',
                    alpha=0.9
                )
            self.pipe_.fit(F_np, y_np)

        elif self.model_type == "lgbm":
            if self.use_optuna:
                best_params = self._optimize_lgbm_params(F_np, y_np)
                self.best_params_ = best_params
                self.pipe_ = lgb.LGBMRegressor(**best_params)
            else:
                self.pipe_ = lgb.LGBMRegressor(
                    n_estimators=self.lgbm_n_estimators,
                    learning_rate=self.lgbm_learning_rate,
                    num_leaves=self.lgbm_num_leaves,
                    min_data_in_leaf=self.lgbm_min_data_in_leaf,
                    feature_fraction=self.lgbm_feature_fraction,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    objective='regression',
                    random_state=self.random_state,
                    verbose=-1
                )
            self.pipe_.fit(F_np, y_np)
            
        elif self.model_type == "rf":
            self.pipe_ = RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.pipe_.fit(F_np, y_np)
            
        elif self.model_type == "nn":
            self.scaler_ = StandardScaler()
            F_scaled = self.scaler_.fit_transform(F_np)
            
            self.pipe_ = MLPRegressor(
                hidden_layer_sizes=self.nn_hidden_layers,
                activation='relu',
                solver='adam',
                learning_rate_init=self.nn_learning_rate,
                max_iter=self.nn_max_iter,
                alpha=self.nn_alpha,
                early_stopping=self.nn_early_stopping,
                validation_fraction=0.1 if self.nn_early_stopping else 0.0,
                random_state=self.random_state,
                verbose=False
            )
            self.pipe_.fit(F_scaled, y_np)
            
        else:  # Ridge with CV
            n_splits = min(self.cv_splits, max(2, len(F) // 200))
            tscv = TimeSeriesSplit(n_splits=n_splits)

            best_alpha, best_mse = None, np.inf
            for a in self.alpha_grid:
                mses = []
                for tr_idx, va_idx in tscv.split(F_np):
                    X_tr, X_va = F_np[tr_idx], F_np[va_idx]
                    y_tr, y_va = y_np[tr_idx], y_np[va_idx]
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

            self.pipe_ = Pipeline([("scaler", StandardScaler()), 
                                  ("model", Ridge(alpha=self.best_alpha_, random_state=self.random_state))])
            self.pipe_.fit(F_np, y_np)

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
        
        if self.model_type == "nn" and self.scaler_ is not None:
            F_scaled = self.scaler_.transform(F.values)
            y_hat = self.pipe_.predict(F_scaled)
        else:
            y_hat = self.pipe_.predict(F.values)
        
        return pd.Series(y_hat, index=F.index, name="y_pred")


