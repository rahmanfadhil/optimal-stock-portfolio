# Optimal stock portfolio

- **Author:** Abdurrahman Fadhil
- **Date:** 26 June 2025

## Abstract

I investigate the out-of-sample performance of three portfolio construction strategies: the naive equally-weighted (1/N) portfolio, the Minimum Variance Portfolio (MVP), and the Maximum Sharpe Ratio Portfolio (MSRP). Using monthly returns data from 12 Indonesian stocks spanning January 2004 to May 2025, I employ a rolling-window backtesting methodology to evaluate their performance across various rebalancing frequencies. I find that the MSRP consistently yields superior risk-adjusted returns and significantly lower maximum drawdowns compared to both the MVP and the 1/N portfolio. The 1/N portfolio often demonstrates competitive absolute returns, even outperforming optimization-based strategies in many scenarios. However, it does so at the cost of substantially higher volatility and deeper drawdowns. The MVP, despite achieving the lowest volatility, severely underperforms in terms of absolute and risk-adjusted returns.

## Usage

1. Create environment
   ```
   $ python -m venv venv
   $ source venv/bin/activate
   ```
2. Install dependencies
   ```
   $ pip install -r requirements.txt
   ```
3. Run the simulation
   ```
   $ python main.py
   ```
