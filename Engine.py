import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import timedelta
import random
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# -------------------------
# INPUTS
# -------------------------
print("--- S&P 500 Alpha Engine (Multi-Period Search) ---")
initial_inv = float(input("Enter investment amount: ") or 10000)
num_simulations = 20  # Number of different time windows to test

FRED_API_KEY = "PASTE_YOUR_API_KEY_HERE"

# -------------------------
# DATA PIPELINE (Large Batch)
# -------------------------
def fetch_master_data():
    # Fetch 20 years of data to allow for random sampling
    full_start = "2004-11-18" # GLD inception
    full_end = "2024-01-01"

    series_map = {"UNRATE": "unemp", "DGS10": "y10", "DTB3": "m3"}
    macro = pd.concat([pdr.DataReader(k, "fred", full_start, full_end, api_key=FRED_API_KEY) for k in series_map.keys()], axis=1)
    macro.columns = series_map.values()
    macro = macro.resample("ME").last().ffill()

    tickers = yf.download(["SPY", "GLD"], start=full_start, end=full_end, auto_adjust=True, progress=False)
    prices = tickers['Close'].resample("ME").last()

    df = pd.concat([macro, prices], axis=1).dropna()
    return df

master_df = fetch_master_data()

def run_strategy(df):
    df = df.copy()
    df["spy_ret"] = df["SPY"].pct_change()
    df["gld_ret"] = df["GLD"].pct_change()
    df["sma"] = df["SPY"].rolling(10).mean()
    df["yc"] = df["y10"] - df["m3"]
    df["u_trend"] = df["unemp"] - df["unemp"].rolling(12).mean()
    df = df.dropna()

    # Strategy Logic
    df["exp_spy"] = 1.0
    df["exp_gld"] = 0.0
    mu, var = df["spy_ret"].rolling(12).mean(), df["spy_ret"].rolling(12).var()
    kelly = ((mu / var) * 0.5).fillna(1.0).clip(1.0, 1.8)

    is_bull = (df["yc"] > 0) & (df["u_trend"] <= 0)
    is_bear = (df["yc"] < 0) | (df["u_trend"] > 0)
    trend_ok = df["SPY"] > df["sma"]

    df.loc[is_bull, "exp_spy"] = kelly
    df.loc[is_bear & trend_ok, ["exp_spy", "exp_gld"]] = [0.8, 0.0]
    df.loc[is_bear & ~trend_ok, ["exp_spy", "exp_gld"]] = [0.0, 1.0]

    df["strat_ret"] = (df["exp_spy"].shift(1) * df["spy_ret"]) + (df["exp_gld"].shift(1) * df["gld_ret"])
    return df.dropna()

def get_metrics(df):
    spy_cum = (1 + df["spy_ret"]).cumprod()
    gld_cum = (1 + df["gld_ret"]).cumprod()
    strat_cum = (1 + df["strat_ret"]).cumprod()

    sharpe = (df["strat_ret"].mean() / df["strat_ret"].std()) * np.sqrt(12) if df["strat_ret"].std() != 0 else 0
    peak = strat_cum.expanding().max()
    mdd = ((strat_cum/peak) - 1).min()

    return {
        "final_val": initial_inv * strat_cum.iloc[-1],
        "spy_val": initial_inv * spy_cum.iloc[-1],
        "gld_val": initial_inv * gld_cum.iloc[-1],
        "spy_pct": (spy_cum.iloc[-1]-1)*100,
        "gld_pct": (gld_cum.iloc[-1]-1)*100,
        "strat_pct": (strat_cum.iloc[-1]-1)*100,
        "sharpe": sharpe,
        "mdd": mdd * 100,
        "spy_sharpe": (df["spy_ret"].mean() / df["spy_ret"].std()) * np.sqrt(12),
        "gld_sharpe": (df["gld_ret"].mean() / df["gld_ret"].std()) * np.sqrt(12),
        "spy_mdd": (((spy_cum/spy_cum.expanding().max())-1).min()) * 100,
        "gld_mdd": (((gld_cum/gld_cum.expanding().max())-1).min()) * 100,
        "avg_exp": df["exp_spy"].mean()
    }

# -------------------------
# SEARCH FOR BEST PERIOD
# -------------------------
best_score = -float('inf')
best_period = None
best_metrics = None

for _ in range(num_simulations):
    # Select random start between 2005 and 2014
    start_idx = random.randint(0, len(master_df) - (12 * 10))
    # Select random duration between 5 and 10 years (60 to 120 months)
    duration = random.randint(60, 120)
    temp_df = master_df.iloc[start_idx : start_idx + duration]

    results_df = run_strategy(temp_df)
    metrics = get_metrics(results_df)

    # Selection criteria: Total strategy return
    if metrics["strat_pct"] > best_score:
        best_score = metrics["strat_pct"]
        best_metrics = metrics
        best_period = (results_df.index[0], results_df.index[-1])

# -------------------------
# FINAL OUTPUT
# -------------------------
print("\n" + "="*80)
print(f"BEST PERFORMANCE PERIOD: {best_period[0].date()} to {best_period[1].date()}")
print("="*80)
col_fmt = "{:<25} | {:<15} | {:<15} | {:<15}"
print(col_fmt.format("Metric", "S&P 500", "Gold Strategy", "Alpha Strategy"))
print("-" * 80)

print(col_fmt.format("Final Portfolio Value",
                     f"${best_metrics['spy_val']:>11,.2f}",
                     f"${best_metrics['gld_val']:>11,.2f}",
                     f"${best_metrics['final_val']:>11,.2f}"))

print(col_fmt.format("Total Percentage",
                     f"{best_metrics['spy_pct']:>13.2f}%",
                     f"{best_metrics['gld_pct']:>13.2f}%",
                     f"{best_metrics['strat_pct']:>13.2f}%"))

print(col_fmt.format("Sharpe Ratio",
                     f"{best_metrics['spy_sharpe']:>14.3f}",
                     f"{best_metrics['gld_sharpe']:>14.3f}",
                     f"{best_metrics['sharpe']:>14.3f}"))

print(col_fmt.format("Max Drawdown",
                     f"{best_metrics['spy_mdd']:>13.2f}%",
                     f"{best_metrics['gld_mdd']:>13.2f}%",
                     f"{best_metrics['mdd']:>13.2f}%"))

print("-" * 80)
alpha = ((best_metrics['final_val'] / best_metrics['spy_val']) - 1) * 100
print(f"STRATEGY OUTPERFORMANCE: {alpha:+.2f}% vs S&P 500")
print(f"Average Exposure Used:   {best_metrics['avg_exp']:.2f}x SPY")
print("="*80)