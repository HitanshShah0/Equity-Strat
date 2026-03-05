import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def run_monte_carlo(ticker, days=252, simulations=1000):
    df = yf.download(ticker, period='2y')
    
    if 'Adj Close' in df.columns:
        data = df['Adj Close']
    else:
        data = df['Close']
    
    # Force to 1D Series and convert to float to prevent dimension errors
    data = data.squeeze().astype(float).dropna()
    returns = data.pct_change().dropna()
    
    mu = returns.mean()           
    sigma = returns.std()         
    last_price = float(data.iloc[-1])
    
    results = np.zeros((days, simulations))
    
    for i in range(simulations):
        shocks = np.random.normal(0, 1, days)
        price_path = [last_price]
        for s in shocks:
            # GBM Formula
            next_price = price_path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * s)
            price_path.append(next_price)
        results[:, i] = price_path[1:]

    return results, last_price, returns

ticker_symbol = "SPY"  
sim_days = 252         
num_sims = 1000

paths, start_price, hist_returns = run_monte_carlo(ticker_symbol, sim_days, num_sims)
final_prices = paths[-1, :]

var_95 = np.percentile(final_prices, 5)
expected_return = (np.mean(final_prices) - start_price) / start_price * 100

plt.figure(figsize=(12, 6))
plt.plot(paths, color='blue', alpha=0.05)
plt.axhline(start_price, color='red', linestyle='--', label=f'Start Price: {start_price:.2f}')
plt.title(f"Monte Carlo Simulation: {num_sims} Paths for {ticker_symbol}")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

print(f"--- Simulation Results for {ticker_symbol} ---")
print(f"Start Price: ${start_price:.2f}")
print(f"Average Final Price: ${np.mean(final_prices):.2f} ({expected_return:.2f}% Expected Return)")
print(f"95% Value at Risk (VaR): ${start_price - var_95:.2f}")
print(f"Worst Case (1%): ${np.min(final_prices):.2f}")