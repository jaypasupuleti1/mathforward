import yfinance as yf
import pandas as pd
import numpy as np


def analyze_investment_math(tickers, initial_investment=1000):
    # 1. DATA ACQUISITION
    # Retrieve a matrix of prices P where P_ij is the price of asset j at time i
    data = yf.download(tickers, period="1y", progress=False)['Close']

    # 2. CALCULATE DAILY RETURNS (Percentage Change)
    # Formula: R_t = (P_t - P_{t-1}) / P_{t-1}
    returns = data.pct_change().dropna()

    stats = pd.DataFrame()

    # 3. EXPECTED ANNUAL RETURN (The First Moment)
    # Formula: E[R_annual] = [ (1/n) * Σ R_i ] * 252
    # We multiply by 252 to annualize the daily mean return.
    stats['Avg Annual Return'] = returns.mean() * 252

    # 4. ANNUAL VOLATILITY (The Second Moment / Standard Deviation)
    # Formula: σ_annual = √[ Σ (R_i - μ)² / (n-1) ] * √252
    # Note: Volatility scales with the square root of time (√T).
    stats['Annual Volatility'] = returns.std() * np.sqrt(252)

    # 5. SHARPE RATIO (The Gradient of the Capital Allocation Line)
    # Formula: S = (μ_p - R_f) / σ_p
    # High S indicates a steeper slope on the Risk-Reward plane.
    stats['Sharpe Ratio'] = stats['Avg Annual Return'] / stats['Annual Volatility']

    # 6. RETURN ON INVESTMENT (Scalar Projection)
    # Formula: V_final = (P_final / P_initial) * V_initial
    start_prices = data.iloc[0]
    end_prices = data.iloc[-1]
    stats['Current Value'] = (end_prices / start_prices) * initial_investment

    # 7. OPTIMIZATION: Argmax of the Sharpe Ratio
    # Find the ticker i that maximizes S_i
    best_ticker = stats['Sharpe Ratio'].idxmax()

    print("\n--- Quantitative Summary ---")
    print(stats.sort_values(by='Sharpe Ratio', ascending=False).to_string())

    print(f"\n★ MATHEMATICAL OPTIMUM: {best_ticker} ★")
    print(f"This asset offers the highest return per unit of risk over the T=1yr interval.")


# Test it
analyze_investment_math(['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA'])