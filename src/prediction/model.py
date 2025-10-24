# src/prediction/model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import joblib
from datetime import timedelta

# This must match the training script
LOOK_BACK_DAYS = 60

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """Loads both the trained LSTM model and its corresponding scaler from file."""
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading scaler from {scaler_path}: {e}")
    return model, scaler

def get_historical_performance(model, scaler, prices_series):
    """Evaluates a model's historical performance on the test set."""
    if model is None or scaler is None:
        return {'error': 'Model or scaler failed to load.'}, pd.DataFrame()

    prices_data = prices_series.values.reshape(-1, 1)
    scaled_data = scaler.transform(prices_data) # Use transform, not fit_transform
    
    train_size = int(len(scaled_data) * 0.8)
    if len(scaled_data) < (train_size + LOOK_BACK_DAYS):
        return {'error': "Not enough data to form a test set."}, pd.DataFrame()

    test_input_data = scaled_data[train_size - LOOK_BACK_DAYS:, :]
    X_test, y_test = [], []
    for i in range(len(test_input_data) - LOOK_BACK_DAYS):
        X_test.append(test_input_data[i:(i + LOOK_BACK_DAYS), 0])
        y_test.append(test_input_data[i + LOOK_BACK_DAYS, 0])
    
    if not X_test:
        return {'error': "Could not create test samples."}, pd.DataFrame()
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_scaled = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(actual_prices, predicted_prices)),
        'mae': mean_absolute_error(actual_prices, predicted_prices),
        'mape': np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-6))) * 100,
        'r2': r2_score(actual_prices, predicted_prices)
    }
    
    plot_df = pd.DataFrame({
        'Actual Price': actual_prices.flatten(), 
        'Predicted Price': predicted_prices.flatten()
    }, index=prices_series.index[-len(actual_prices):])
    
    return metrics, plot_df

def predict_future_price(model, scaler, prices_series, target_date):
    """Predicts the price for a single, specific date."""
    if model is None or scaler is None:
        return {'error': 'Model or scaler failed to load.'}
    
    last_known_date = target_date - timedelta(days=1)
    input_data = prices_series.loc[:last_known_date].tail(LOOK_BACK_DAYS)
    
    if len(input_data) < LOOK_BACK_DAYS:
        return {'error': f"Not enough historical data before {target_date.date()} to predict. Need {LOOK_BACK_DAYS} days."}

    scaled_input = scaler.transform(input_data.values.reshape(-1, 1))
    X_predict = np.reshape(scaled_input, (1, LOOK_BACK_DAYS, 1))
    
    predicted_scaled_price = model.predict(X_predict)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)[0, 0]
    
    try:
        actual_price = prices_series.loc[target_date]
    except KeyError:
        actual_price = "Not Available (Future Date)"

    return {
        'predicted_price': predicted_price,
        'actual_price': actual_price,
        'last_known_price': input_data.iloc[-1]
    }