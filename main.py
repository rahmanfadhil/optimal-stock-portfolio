import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configuration Parameters
LOOKBACK_PERIOD = 60  # Lookback period in months
TRANSACTION_COST_RATE = 0.0025  # 0.25%
RISK_FREE_RATE_MONTHLY = 0.025 / 12  # 2.5% annual rate divided by 12
MAXIMUM_EXPOSURE = 0.4 # 40% maximum exposure to a single stock

# Ensure output folder exists
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class OptimizationFailureException(Exception):
    """Custom exception for optimization failures."""
    def __init__(self, strategy: str, error_message: str):
        super().__init__(f"{strategy} optimization failed: {error_message}")

def load_and_prepare_data(file_path='data.csv'):
    """Loads and prepares stock data for backtesting."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values(by=['Ticker', 'Date'])
    df['PrevPrice'] = df.groupby('Ticker')['Price'].shift(1)
    df['MonthlyReturn'] = (df['Price'] + df['Dividends'] - df['PrevPrice']) / df['PrevPrice']
    returns_df = df.pivot(index='Date', columns='Ticker', values='MonthlyReturn')
    start_date_sim = pd.to_datetime('2004-01-01')
    end_date_sim = pd.to_datetime('2025-05-31')
    returns_df = returns_df[(returns_df.index >= start_date_sim) & (returns_df.index <= end_date_sim)]
    returns_df = returns_df.dropna()
    return returns_df

def get_1n_weights(num_assets: int) -> np.ndarray:
    """Calculates weights for the Equally Weighted (1/N) portfolio."""
    return np.ones(num_assets) / num_assets

def get_mvp_weights(exp_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Calculates weights for the Minimum Variance Portfolio (MVP)."""
    num_assets = len(exp_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, MAXIMUM_EXPOSURE) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    def objective_function(weights: np.ndarray, cov: np.ndarray) -> float:
        return np.dot(weights.T, np.dot(cov, weights))

    if np.linalg.cond(cov_matrix) > 1e10:
        logging.warning("Covariance matrix is ill-conditioned for MVP. Adding diagonal for stability.")
        cov_matrix = cov_matrix + np.eye(num_assets) * 1e-6

    result = minimize(objective_function, initial_weights, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        raise OptimizationFailureException("Minimum-Variance Portfolio (MVP)", result.message)

def get_msrp_weights(exp_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> np.ndarray:
    """Calculates weights for the Maximum Sharpe Ratio Portfolio (MSRP)."""
    num_assets = len(exp_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, MAXIMUM_EXPOSURE) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    def objective_function(weights: np.ndarray, exp_ret: np.ndarray, cov: np.ndarray, rf: float) -> float:
        portfolio_return = np.dot(weights, exp_ret)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        if portfolio_volatility == 0:
            return 1e10
        return -(portfolio_return - rf) / portfolio_volatility

    if np.linalg.cond(cov_matrix) > 1e10:
        logging.warning("Covariance matrix is ill-conditioned for MSRP. Adding diagonal for stability.")
        cov_matrix = cov_matrix + np.eye(num_assets) * 1e-6

    result = minimize(objective_function, initial_weights, args=(exp_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        raise OptimizationFailureException("Maximize Sharpe Ratio Portfolio (MSRP)", result.message)

def calculate_max_drawdown(portfolio_values_df: pd.DataFrame) -> float:
    """Calculates the maximum drawdown from portfolio value history."""
    if portfolio_values_df.empty:
        return np.nan
    values = portfolio_values_df['Value'].values
    peak = values[0]
    max_drawdown = 0
    for value in values:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    return max_drawdown

class Portfolio:
    """Manages portfolio state during backtesting."""
    def __init__(self, initial_investment: float, num_assets: int):
        self.initial_investment = initial_investment
        self.current_value = initial_investment
        self.current_weights = np.ones(num_assets) / num_assets
        self.portfolio_values_history = []
        self.portfolio_weights_at_start_of_month = []
        self.num_assets = num_assets
        self.total_transaction_costs = 0

    def _calculate_transaction_cost(self, actual_weights_before_rebalance: np.ndarray,
                                   target_weights: np.ndarray, transaction_cost_rate: float) -> float:
        """Calculates transaction cost based on portfolio turnover."""
        turnover = np.sum(np.abs(target_weights - actual_weights_before_rebalance))
        return self.current_value * turnover * transaction_cost_rate

    def rebalance(self, current_date: pd.Timestamp, lookback_data: pd.DataFrame,
                  strategy_name: str, transaction_cost_rate: float,
                  risk_free_rate: float, is_initial_rebalance: bool) -> None:
        """Rebalances the portfolio according to the strategy."""
        exp_returns = lookback_data.mean().values
        cov_matrix = lookback_data.cov().values
        if is_initial_rebalance:
            actual_weights_before_rebalance = np.zeros(self.num_assets)  # Fixed for initial allocation
        else:
            actual_weights_before_rebalance = self.current_weights

        if strategy_name == '1/N':
            target_weights = get_1n_weights(self.num_assets)
        elif strategy_name == 'MVP':
            target_weights = get_mvp_weights(exp_returns, cov_matrix)
        elif strategy_name == 'MSRP':
            target_weights = get_msrp_weights(exp_returns, cov_matrix, risk_free_rate)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        cost = self._calculate_transaction_cost(actual_weights_before_rebalance, target_weights, transaction_cost_rate)
        self.current_value -= cost
        self.total_transaction_costs += cost
        logging.debug(f"[{current_date.strftime('%Y-%m-%d')}] Rebalancing. Cost: ${cost:,.2f}")
        self.current_weights = target_weights.copy()
        self.portfolio_weights_at_start_of_month.append({'Date': current_date, 'Weights': self.current_weights.tolist()})

    def apply_monthly_returns(self, current_date: pd.Timestamp, monthly_returns_for_assets: np.ndarray) -> None:
        """Applies monthly returns to the portfolio."""
        monthly_portfolio_return = np.dot(self.current_weights, monthly_returns_for_assets)
        self.current_value *= (1 + monthly_portfolio_return)
        self.portfolio_values_history.append({'Date': current_date, 'Value': self.current_value})
        self.current_weights = self.current_weights * (1 + monthly_returns_for_assets)
        self.current_weights = self.current_weights / np.sum(self.current_weights)

def run_backtest(returns_df: pd.DataFrame, strategy_name: str, lookback_period: int,
                 rebalancing_frequency: int, transaction_cost_rate: float,
                 risk_free_rate: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Simulates a portfolio strategy over time."""
    num_assets = returns_df.shape[1]
    portfolio = Portfolio(initial_investment=10_000_000, num_assets=num_assets)
    for i in range(lookback_period, len(returns_df)):
        current_date = returns_df.index[i]
        monthly_returns_for_assets = returns_df.iloc[i].values
        is_rebalance_point = (i == lookback_period) or ((i - lookback_period) % rebalancing_frequency == 0)
        if is_rebalance_point:
            lookback_data = returns_df.iloc[i - lookback_period : i]
            portfolio.rebalance(current_date, lookback_data, strategy_name,
                                transaction_cost_rate, risk_free_rate, is_initial_rebalance=(i == lookback_period))
        portfolio.apply_monthly_returns(current_date, monthly_returns_for_assets)
    return pd.DataFrame(portfolio.portfolio_values_history), \
           pd.DataFrame(portfolio.portfolio_weights_at_start_of_month), \
           portfolio.total_transaction_costs

def main():
    """Runs the backtest and displays results."""
    returns_data = load_and_prepare_data()
    logging.info("--- Data Information ---")
    logging.info("Returns data head (first 5 rows):\n%s", returns_data.head())
    logging.info(f"\nSimulation Period: {returns_data.index.min().strftime('%Y-%m-%d')} to {returns_data.index.max().strftime('%Y-%m-%d')}")
    logging.info(f"Number of assets: {returns_data.shape[1]}")
    logging.info(f"Total months in data: {len(returns_data)}")

    # Data Summary Statistics
    logging.info("\n--- Data Summary Statistics ---")

    # General information
    num_stocks = returns_data.shape[1]
    num_months = returns_data.shape[0]
    start_date = returns_data.index.min().strftime('%Y-%m-%d')
    end_date = returns_data.index.max().strftime('%Y-%m-%d')
    logging.info(f"Number of stocks: {num_stocks}")
    logging.info(f"Number of monthly observations: {num_months}")
    logging.info(f"Date range: {start_date} to {end_date}")

    # Summary statistics for each stock
    summary_stats = pd.DataFrame({
        'Mean Return (%)': returns_data.mean() * 100,
        'Std Dev (%)': returns_data.std() * 100,
        'Min Return (%)': returns_data.min() * 100,
        'Max Return (%)': returns_data.max() * 100,
        'Skewness': returns_data.skew(),
        'Kurtosis': returns_data.kurtosis(),
        'Observations': returns_data.count()
    }).round(2)
    logging.info("\nSummary Statistics for Each Stock (Monthly Returns):")
    logging.info("\n%s", summary_stats.to_string())
    summary_stats.to_csv("output/summary_stats.csv")

    # Correlation matrix
    corr_matrix = returns_data.corr().round(2)
    logging.info("\nFull-Sample Correlation Matrix (Monthly Returns):")
    logging.info("\n%s", corr_matrix.to_string())
    corr_matrix.to_csv("output/corr_matrix.csv")

    if len(returns_data) < LOOKBACK_PERIOD:
        logging.error(f"Not enough data for a {LOOKBACK_PERIOD}-month lookback. Available: {len(returns_data)} months.")
        return

    strategies = ['1/N', 'MVP', 'MSRP']
    rebalancing_frequencies = [1, 2, 3, 6, 12]
    results = []

    for strategy in strategies:
        for freq in rebalancing_frequencies:
            logging.info(f"\n--- Running backtest for {strategy} strategy with rebalancing frequency {freq} month(s) ---")
            portfolio_values_df, _, total_transaction_costs = run_backtest(
                returns_data, strategy, LOOKBACK_PERIOD, freq, TRANSACTION_COST_RATE, RISK_FREE_RATE_MONTHLY
            )
            if not portfolio_values_df.empty:
                df_values = portfolio_values_df.copy()
                df_values['MonthlyPortfolioReturn'] = df_values['Value'].pct_change()
                cumulative_return = (df_values['Value'].iloc[-1] / df_values['Value'].iloc[0]) - 1
                num_months_simulated = len(df_values) - 1
                if num_months_simulated > 0:
                    annualized_return = (1 + cumulative_return)**(12/num_months_simulated) - 1
                    annualized_volatility = df_values['MonthlyPortfolioReturn'].std() * np.sqrt(12)
                    sharpe_ratio = (annualized_return - (RISK_FREE_RATE_MONTHLY * 12)) / annualized_volatility if annualized_volatility != 0 else np.nan
                else:
                    annualized_return, annualized_volatility, sharpe_ratio = np.nan, np.nan, np.nan
                max_drawdown = calculate_max_drawdown(df_values)
                results.append({
                    'Strategy': strategy,
                    'Rebalancing Frequency (Months)': freq,
                    'Final Value (Million IDR)': df_values['Value'].iloc[-1] / 1_000_000,
                    'Annualized Return (%)': annualized_return * 100,
                    'Annualized Volatility (%)': annualized_volatility * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    'Maximum Drawdown (%)': max_drawdown * 100,
                    'Total Transaction Costs (Million IDR)': total_transaction_costs / 1_000_000
                })
            else:
                logging.warning(f"No results for {strategy} with frequency {freq} months.")

    # Create and display results table
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index(['Strategy', 'Rebalancing Frequency (Months)'])
    results_df = results_df.round(2)
    logging.info("\n--- Performance Metrics Table ---")
    logging.info("\n%s", results_df.to_string())
    results_df.to_csv("output/simulation.csv")

if __name__ == '__main__':
    main()