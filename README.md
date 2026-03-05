# Equity-Strat
Developed a quantitative backtesting engine in Python that uses the Kelly Criterion and FRED macroeconomic data to outperform the S&amp;P 500 by X% in historical simulations.
# AlphaEngine: Macro-Driven Tactical Asset Allocation

AlphaEngine is a quantitative backtesting engine that utilizes macroeconomic indicators (Yield Curve, Unemployment Trends) and technical signals (SMA) to dynamically rotate between S&P 500 (SPY) and Gold (GLD).

## Key Features
* **Macro Integration:** Uses FRED API to pull 10Y-3M Yield Curve and Unemployment data.
* **Kelly Criterion Scaling:** Dynamically adjusts position sizing based on rolling volatility and returns.
* **Multi-Period Search:** Runs Monte Carlo-style simulations across random historical windows to identify strategy robustness.
* **Performance Metrics:** Calculates Sharpe Ratio, Maximum Drawdown (MDD), and Alpha vs. S&P 500.

## Strategy Logic
The algorithm switches regimes based on three primary signals:
1. **Bull Regime:** (Yield Curve > 0 & Unemployment trending down) -> Leveraged SPY.
2. **Bear/Hedged Regime:** (Yield Curve < 0 or Unemployment rising) -> 
   - If SPY > 10-month SMA: Hold 80% SPY.
   - If SPY < 10-month SMA: Rotate 100% to Gold (GLD).



## Tech Stack
* **Python 3.x**
* **Pandas/NumPy:** Data manipulation and vectorization.
* **yFinance & Pandas_DataReader:** Financial data ingestion.
* **Matplotlib:** (Optional) Performance visualization.

## Setup & Usage
1. Clone the repo: `git clone https://github.com/yourusername/alpha-engine.git`
2. Install dependencies: `pip install pandas numpy yfinance pandas_datareader`
3. Get a FREE API Key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
4. Run the engine: `python alpha_engine.py`
