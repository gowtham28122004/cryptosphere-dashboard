# src/analysis/portfolio_math.py
import pandas as pd
import numpy as np

def assign_weights(investment_goal, assets, daily_returns_df=None):
    """Assigns portfolio weights based on various strategies."""
    weights = {}
    if investment_goal == 'risk_parity':
        if daily_returns_df is None: raise ValueError("daily_returns_df is required for 'risk_parity'.")
        volatilities = daily_returns_df[assets].std()
        inverse_volatilities = 1 / (volatilities + 1e-6)
        weights = (inverse_volatilities / inverse_volatilities.sum()).to_dict()
    elif investment_goal == 'sharpe_max':
        if daily_returns_df is None: raise ValueError("daily_returns_df is required for 'sharpe_max'.")
        mean_returns = daily_returns_df[assets].mean()
        volatilities = daily_returns_df[assets].std()
        sharpe_ratios = mean_returns / (volatilities + 1e-6)
        positive_sharpe = sharpe_ratios.clip(lower=0)
        weights = (positive_sharpe / positive_sharpe.sum()).to_dict()
    elif investment_goal == 'momentum':
        if daily_returns_df is None: raise ValueError("daily_returns_df is required for 'momentum'.")
        total_returns = (1 + daily_returns_df[assets]).prod() - 1
        positive_momentum = total_returns.clip(lower=0)
        weights = (positive_momentum / positive_momentum.sum()).to_dict()
    elif investment_goal == 'risk_level':
        weights = {'BTCUSDT': 0.40, 'ETHUSDT': 0.30, 'SOLUSDT': 0.12, 'ADAUSDT': 0.06, 'XRPUSDT': 0.06, 'DOGEUSDT': 0.06}
    elif investment_goal == 'market_cap':
        weights = {'BTCUSDT': 0.50, 'ETHUSDT': 0.3333, 'SOLUSDT': 0.0556, 'ADAUSDT': 0.0444, 'XRPUSDT': 0.0444, 'DOGEUSDT': 0.0222}
    elif investment_goal == 'safety':
        weights = {'BTCUSDT': 0.50, 'ETHUSDT': 0.3333, 'SOLUSDT': 0.0833, 'ADAUSDT': 0.05, 'XRPUSDT': 0.0333, 'DOGEUSDT': 0.00}
    elif investment_goal == 'growth':
        weights = {'BTCUSDT': 0.20, 'ETHUSDT': 0.40, 'SOLUSDT': 0.20, 'ADAUSDT': 0.10, 'XRPUSDT': 0.05, 'DOGEUSDT': 0.05}
    elif investment_goal == 'equal':
        equal_weight = 1 / len(assets)
        for asset in assets: weights[asset] = round(equal_weight, 4)
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        weights = {asset: w / total_weight for asset, w in weights.items()}
    return weights

def calculate_portfolio_return(weights, daily_returns_df):
    """Calculates the historical daily returns of a portfolio."""
    filtered_weights = {k: v for k, v in weights.items() if k in daily_returns_df.columns}
    weights_series = pd.Series(filtered_weights).reindex(daily_returns_df.columns, fill_value=0)
    return daily_returns_df.mul(weights_series, axis='columns').sum(axis=1)