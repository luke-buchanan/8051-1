import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class Student:
    """Ultra-simple baseline for debugging - just 5 lagged returns"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.scaler = None
        self.fitted_ = False
    
    def fit(self, X_train, y_train, meta=None):
        # Ultra-simple: just use last 5 returns
        close = X_train['Close'] if 'Close' in X_train.columns else X_train.iloc[:, 0]
        lr = np.log(close / close.shift(1))
        
        F = pd.DataFrame({
            'lag1': lr.shift(1),
            'lag2': lr.shift(2),
            'lag3': lr.shift(3),
            'lag4': lr.shift(4),
            'lag5': lr.shift(5),
        }).dropna()
        
        y = y_train.reindex(F.index).dropna()
        F = F.loc[y.index]
        
        print(f"Fit: {len(F)} samples, y mean={y.mean():.6f}, y>0: {(y>0).sum()}, y<0: {(y<0).sum()}")
        
        if len(F) < 50:
            self.model = None
            self.fitted_ = True
            return self
        
        self.scaler = StandardScaler()
        F_scaled = self.scaler.fit_transform(F.values)
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(F_scaled, y.values)
        self.fitted_ = True
        return self
    
    def predict(self, X, meta=None):
        if not self.fitted_ or self.model is None:
            return pd.Series(0.0, index=X.index, name="y_pred")
        
        close = X['Close'] if 'Close' in X.columns else X.iloc[:, 0]
        lr = np.log(close / close.shift(1))
        
        F = pd.DataFrame({
            'lag1': lr.shift(1),
            'lag2': lr.shift(2),
            'lag3': lr.shift(3),
            'lag4': lr.shift(4),
            'lag5': lr.shift(5),
        }).dropna()
        
        if F.empty:
            return pd.Series(0.0, index=X.index, name="y_pred")
        
        F_scaled = self.scaler.transform(F.values)
        y_hat = self.model.predict(F_scaled)
        return pd.Series(y_hat, index=F.index, name="y_pred")