# Optimal stock portfolio

- **Author:** Abdurrahman Fadhil
- **Date:** 25 June 2025

## Abstract

This paper evaluates the practical performance of naive implementations of the Markowitz (1952) Mean-Variance Optimization (MVO) framework on a portfolio of 12 Indonesian stocks from 2009 to 2025. I construct and backtest two MVO-based strategies—Minimum Variance and Maximum Sharpe Ratio—and compare their performance against the equally weighted (1/N) portfolio benchmark. The MVO inputs are estimated using simple historical sample averages from a rolling 60-month lookback window. The strategies are simulated accounting for various rebalancing frequencies from monthly to annually, incorporating transaction costs. The results indicate that while the Maximum Sharpe Ratio portfolio achieves the highest risk-adjusted returns (Sharpe Ratio), the simple 1/N portfolio generates the highest terminal wealth, especially when rebalanced infrequently. The Minimum Variance portfolio successfully reduces volatility but at the cost of significantly lower returns. The findings underscore the formidable challenge of estimation error in practical applications of MVO and reaffirm the robustness of the 1/N heuristic.

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
