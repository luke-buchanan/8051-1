# mltester.py
"""
mltester: minimal harness to evaluate single-stock Student via walk-forward.

(1) Staff responsibilities: build the target y = log(C_{t+H}/C_t), run a leakage-safe
    expanding-window evaluation, compute DirAcc/MAE/RMSE, and save results.
(2) Student responsibilities: all feature engineering + model selection live inside
    Student; mltester does not compute features.

API expected from Student:
  (1) fit(X_train: pd.DataFrame, y_train: pd.Series, meta: dict|None) -> self
  (2) predict(X: pd.DataFrame, meta: dict|None) -> pd.Series named 'y_pred'
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd


# ============================================================
# Loading a Student class from "module_or_path:ClassName"
# ============================================================

def load_class(spec: str) -> Type:
    """
    Load a class from either an importable module or a .py file path.

    Examples:
      "my_pkg.student_model:Student"
      "./student.py:Student"
    """
    if ":" not in spec:
        raise ValueError("model spec must be 'module_or_path:ClassName'")
    mod, cls_name = spec.split(":", 1)

    if mod.endswith(".py"):
        mod_path = Path(mod).expanduser().resolve()
        if not mod_path.exists():
            raise FileNotFoundError(f"No such file: {mod_path}")
        module_name = mod_path.stem
        spec_obj = importlib.util.spec_from_file_location(module_name, str(mod_path))
        module = importlib.util.module_from_spec(spec_obj)
        assert spec_obj and spec_obj.loader
        spec_obj.loader.exec_module(module)  # type: ignore[attr-defined]
    else:
        module = importlib.import_module(mod)

    if not hasattr(module, cls_name):
        raise AttributeError(f"Class '{cls_name}' not found in {module}")
    return getattr(module, cls_name)

# ==========================================
# Data I/O helpers (single long file store)
# ==========================================

def read_store(path: Path) -> pd.DataFrame:
    """
    Read a long-format store with at least columns: date, ticker, close.
    Optional columns (open, high, low, volume, adj_close) are passed through.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported data file; use .csv or .parquet")

    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("date", "ticker", "close"):
        if col not in df.columns:
            raise ValueError("Input must contain columns: date, ticker, close")

    df["date"] = pd.to_datetime(df["date"], utc=False)
    # keep base + pass-through if present
    cols = ["date", "ticker", "close"]
    for extra in ("open", "high", "low", "adj_close", "volume"):
        if extra in df.columns:
            cols.append(extra)
    return df[cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def price_frame_from_store(store: pd.DataFrame, ticker: str,
                           start: Optional[str] = None,
                           end: Optional[str] = None) -> pd.DataFrame:
    """
    Return a per-ticker DataFrame indexed by date with at least 'Close' column.
    Passes through any other columns if present (e.g., High/Low/Volume).
    """
    sl = store[store["ticker"] == ticker]
    if sl.empty:
        raise ValueError(f"No rows for ticker {ticker!r} in data store.")

    if start:
        sl = sl[sl["date"] >= pd.Timestamp(start)]
    if end:
        sl = sl[sl["date"] <= pd.Timestamp(end)]
    if sl.empty:
        raise ValueError(f"No data for {ticker} in the requested window.")

    sl = sl.set_index("date").sort_index()
    cols_map = {c: c.capitalize() for c in sl.columns if c != "ticker"}
    pf = sl.drop(columns=["ticker"]).rename(columns=cols_map)
    if "Close" not in pf.columns:
        first = pf.columns[0]
        pf = pf.rename(columns={first: "Close"})
    return pf


# ==============================
# Targets & walk-forward driver
# ==============================

def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """
    y_t = log(C_{t+H} / C_t). NaNs at the tail where future isn't available.
    """
    close = pd.Series(close).astype(float)
    return np.log(close.shift(-horizon) / close)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
    """
    Return (DirAcc, MAE, RMSE) on overlapping, finite indices.
    """
    df = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return (float("nan"), float("nan"), float("nan"))
    diracc = float((np.sign(df["y"]) == np.sign(df["yhat"])).mean())
    mae = float(np.abs(df["y"] - df["yhat"]).mean())
    rmse = float(np.sqrt(((df["y"] - df["yhat"]) ** 2).mean()))
    return diracc, mae, rmse


def walk_forward_predict(
    ModelClass: Type,
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    step: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Leakage-safe expanding-window evaluation.

    - At each refit point, fit on ALL data strictly before the first test day
      in the block; target is next-H-day cumulative log return.
    - Predict for the next `step` test days; keep only those test dates.

    Returns y_true, y_pred (both indexed by date, named 'y_true'/'y_pred').
    """
    if "Close" not in prices.columns:
        raise ValueError("prices must contain a 'Close' column")

    # Build full target once
    y_full = forward_log_return(prices["Close"], horizon=horizon)
    y_full.name = "y_true"
    test_dates = y_full.dropna().index

    preds = []
    for i in range(0, len(test_dates), step):
        block = test_dates[i : i + step]
        first_test = block[0]

        X_train = prices.loc[: first_test - pd.Timedelta(days=1)]
        y_train = y_full.loc[: first_test - pd.Timedelta(days=1)]

        # fresh model each block; accept config/random_state if available
        try:
            model = ModelClass(config=None, random_state=42)
        except TypeError:
            try:
                model = ModelClass(random_state=42)
            except TypeError:
                model = ModelClass()

        model.fit(X_train, y_train, meta={"horizon": horizon})

        X_pred = prices.loc[: block[-1]]
        y_hat = model.predict(X_pred, meta={"horizon": horizon})

        preds.append(y_hat.reindex(block).dropna())

    y_pred = pd.concat(preds) if preds else pd.Series(dtype=float, name="y_pred")
    y_pred.name = "y_pred"
    y_true = y_full.reindex(y_pred.index)
    y_true.name = "y_true"
    return y_true, y_pred


# ====================
# Public test runner
# ====================

def run_mltester(
    model_spec: str,
    tickers: List[str],
    data_file: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    horizon: int = 1,
    step: int = 5,
    out_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Load the model class, run walk-forward per ticker, compute metrics,
    save per-ticker CSVs and a summary DataFrame (also returned).
    """
    store = read_store(Path(data_file))
    ModelClass = load_class(model_spec)

    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tickers:
        pf = price_frame_from_store(store, t, start, end)
        y_true, y_pred = walk_forward_predict(
            ModelClass, pf, horizon=horizon, step=step
        )
        diracc, mae, rmse = compute_metrics(y_true, y_pred)
        pd.concat([y_true, y_pred], axis=1).to_csv(outp / f"mltester_{t}.csv", index_label="date")
        print(f"{t:>6}: DirAcc={diracc:0.4f}  MAE={mae:0.6f}  RMSE={rmse:0.6f}")
        rows.append({"ticker": t, "diracc": diracc, "mae": mae, "rmse": rmse})

    summary = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    if not summary.empty:
        mean_row = {"ticker": "MEAN",
                    "diracc": float(summary["diracc"].mean()),
                    "mae": float(summary["mae"].mean()),
                    "rmse": float(summary["rmse"].mean())}
        summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
        
        # Print the averages
        print("-" * 50)
        print(f"{'MEAN':>6}: DirAcc={mean_row['diracc']:0.4f}  MAE={mean_row['mae']:0.6f}  RMSE={mean_row['rmse']:0.6f}")

    summary.to_csv(outp / "mltester_summary.csv", index=False)
    print(f"Saved outputs to {outp.resolve()}")
    return summary


# =========
#   CLI
# =========

def main():
    ap = argparse.ArgumentParser(description="Student-friendly ML tester (single-stock, walk-forward)")
    ap.add_argument("--model", required=True, help="module_or_path:ClassName (e.g., ./student.py:Student)")
    ap.add_argument("--tickers", nargs="+", required=True, help="Tickers to evaluate")
    ap.add_argument("--data-file", required=True, help="Long file with date,ticker,close[, other columns]")
    ap.add_argument("--start", help="YYYY-MM-DD (optional)")
    ap.add_argument("--end", help="YYYY-MM-DD (optional)")
    ap.add_argument("--horizon", type=int, default=1, help="Prediction horizon H (days)")
    ap.add_argument("--step", type=int, default=5, help="Re-fit cadence (test days per block)")
    ap.add_argument("--out-dir", default="outputs", help="Where to save per-ticker CSVs and summary")
    args = ap.parse_args()

    run_mltester(
        model_spec=args.model,
        tickers=args.tickers,
        data_file=args.data_file,
        start=args.start, end=args.end,
        horizon=args.horizon, step=args.step,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
