# src/analysis/risk_checker.py
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import the assign_weights function to be used here
from .portfolio_math import assign_weights

def run_risk_check_logic(target_to_check, returns_df, all_coins, all_rules):
    """
    Performs the 6-rule risk check on a user-selected portfolio or single coin.
    This version restores the missing Sortino Ratio calculation.
    """
    RISK_THRESHOLDS = {'Volatility': 5.0, 'Sharpe Ratio': 1.0, 'Max Drawdown': -20.0, 'Sortino Ratio': 1.0, 'Beta': 1.2, 'Max Asset Weight': 0.40}
    market_returns = returns_df['BTCUSDT'].dropna()
    metrics = {}
    
    if target_to_check in all_rules:
        weights = assign_weights(target_to_check, all_coins, daily_returns_df=returns_df/100)
        asset_returns = (returns_df[list(weights.keys())] * pd.Series(weights)).sum(axis=1).dropna()
        metrics['Max Asset Weight'] = max(weights.values())
    elif target_to_check in all_coins:
        asset_returns = returns_df[target_to_check].dropna()
        metrics['Max Asset Weight'] = 1.0
    else:
        return pd.DataFrame([{'Error': 'Invalid target selected'}])

    common_index = asset_returns.index.intersection(market_returns.index)
    
    # --- THIS IS THE RESTORED AND CORRECTED METRICS CALCULATION ---
    metrics['Volatility'] = asset_returns.std()
    metrics['Sharpe Ratio'] = (asset_returns.mean() / asset_returns.std()) * np.sqrt(365) if asset_returns.std() != 0 else 0
    metrics['Max Drawdown'] = (((1 + asset_returns/100).cumprod()).cummax() - (1 + asset_returns/100).cumprod()).max() * -100
    
    # Calculate Sortino Ratio
    negative_returns = asset_returns[asset_returns < 0]
    downside_deviation = negative_returns.std()
    if np.isnan(downside_deviation) or downside_deviation == 0:
        metrics['Sortino Ratio'] = np.inf
    else:
        metrics['Sortino Ratio'] = (asset_returns.mean() / downside_deviation) * np.sqrt(365)
        
    metrics['Beta'] = np.cov(asset_returns[common_index], market_returns[common_index])[0,1] / np.var(market_returns[common_index])
    # -----------------------------------------------------------------

    results = []
    for rule, value in metrics.items():
        op = '<=' if rule in ['Volatility', 'Beta', 'Max Asset Weight'] else '>='
        status = 'N/A'
        if rule == 'Max Asset Weight' and target_to_check in all_coins:
            status = 'N/A'
        else:
            passed = eval(f"value {op} RISK_THRESHOLDS[rule]")
            status = '✅ PASS' if passed else '❌ FAIL'
        results.append({'Rule': rule, 'Value': round(value, 4), 'Threshold': f"{op} {RISK_THRESHOLDS[rule]}", 'Status': status})
        
    return pd.DataFrame(results)

def send_email_alert(failed_rules, sender_email, sender_password, receiver_email):
    """Formats and sends an email alert to the specified receiver."""
    subject = f"ALERT: Risk Violation Detected for {failed_rules.iloc[0]['portfolio_name']}"
    body = f"Hello,\n\nThe following risk rules have failed for your monitored asset/portfolio:\n\n"
    for index, failure in failed_rules.iterrows():
        body += (
            f"  - Rule: {failure['Rule']}\n"
            f"    Condition: {failure['Threshold']}\n"
            f"    Actual Value: {failure['Value']}\n\n"
        )
    body += "Please review your portfolio for necessary adjustments.\n"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)