import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os
import joblib

# Import helper functions from your src directory
from src.analysis.portfolio_math import assign_weights, calculate_portfolio_return

# --- Configuration ---
DATABASE_FILE_PATH = 'data/crypto_data.db'
MODELS_DIR = 'models/'
COINS_TO_TRAIN = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT']
PORTFOLIOS_TO_TRAIN = ['risk_level', 'market_cap', 'safety', 'growth', 'equal', 'risk_parity', 'sharpe_max', 'momentum']

LOOK_BACK_DAYS = 60

# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def prepare_target_series(prices_df, returns_df, target_name):
    if target_name in prices_df.columns:
        return pd.DataFrame(prices_df[target_name]).rename(columns={target_name: 'Close'})
    else:
        try:
            weights = assign_weights(target_name, list(prices_df.columns), daily_returns_df=returns_df)
            portfolio_returns = calculate_portfolio_return(weights, returns_df)
            return pd.DataFrame(100 * (1 + portfolio_returns).cumprod(), columns=['Close']).dropna()
        except: return None

def create_lstm_datasets(df):
    prices_data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices_data)
    
    def _create_sequences(dataset, look_back=LOOK_BACK_DAYS):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0]); y.append(dataset[i + look_back, 0])
        X, y = np.array(X), np.array(y)
        return np.reshape(X, (X.shape[0], X.shape[1], 1)), y
        
    X, y = _create_sequences(scaled_data)
    return X, y, scaler

# ==============================================================================
# HYPERPARAMETER TUNING ENGINE
# ==============================================================================
def create_lstm_model(units=50, dropout_rate=0.2, learning_rate=0.001):
    """Builds the Keras model with the modern Input layer syntax."""
    model = Sequential([
        # FIX: Using the recommended Input layer syntax
        Input(shape=(LOOK_BACK_DAYS, 1)),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    print("--- STARTING MODEL TRAINING FACTORY WITH CROSS-VALIDATION ---")
    
    conn = sqlite3.connect(DATABASE_FILE_PATH)
    try:
        df = pd.read_sql_query("SELECT Date, Symbol, Close FROM crypto_prices ORDER BY Date ASC", conn, parse_dates=['Date'])
    except Exception as e:
        print(f"FATAL ERROR: Could not read from database. Error: {e}"); exit()
    finally:
        conn.close()
    
    prices_df = df.pivot(index='Date', columns='Symbol', values='Close')
    returns_df = prices_df.pct_change()
    master_target_list = COINS_TO_TRAIN + PORTFOLIOS_TO_TRAIN

    for target in master_target_list:
        print("\n" + "="*50 + f"\nPROCESSING TARGET: {target}\n" + "="*50)
        target_series_df = prepare_target_series(prices_df, returns_df, target)
        if target_series_df is None or target_series_df.empty or len(target_series_df) < 150:
            print(f"  > Insufficient data for '{target}'. Skipping."); continue
            
        X, y, scaler = create_lstm_datasets(target_series_df)
        if len(X) < 50:
            print(f"  > Not enough data for cross-validation for '{target}'. Skipping."); continue

        print("--- 1. Starting Hyperparameter Search with TimeSeriesSplit CV ---")
        
        # FIX: The KerasRegressor is now correctly configured. The loss is part of the model itself.
        model = KerasRegressor(model=create_lstm_model, verbose=0)
        
        param_grid = {
            'model__units': [50, 75],
            'model__dropout_rate': [0.2],
            'batch_size': [32, 64],
            'epochs': [10],
            'model__learning_rate': [0.001]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        # FIX: The scoring parameter is now correct.
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error')
        
        # The fit call itself can be complex, let's add error handling
        try:
            grid_result = grid_search.fit(X, y)
        except Exception as e:
            print(f"  > ERROR during GridSearchCV fit for '{target}': {e}")
            continue

        print("\n--- Hyperparameter Search Complete ---")
        print(f"Best Score (Negative MSE): {grid_result.best_score_:.6f}")
        print("Best Hyperparameters Found:")
        best_params = grid_result.best_params_
        for param, value in best_params.items():
            print(f"  - {param.replace('model__', '')}: {value}")
        
        print("\n--- 2. Training the Final, Best Model ---")
        
        final_model = create_lstm_model(
            units=best_params['model__units'],
            dropout_rate=best_params['model__dropout_rate'],
            learning_rate=best_params['model__learning_rate']
        )
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
        final_model.fit(X, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1, callbacks=[early_stopping])
        
        print("\n--- 3. Saving the Final Model and Scaler ---")
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{target}_model.keras"); final_model.save(model_path)
        scaler_path = os.path.join(MODELS_DIR, f"{target}_scaler.joblib"); joblib.dump(scaler, scaler_path)
        
        print(f"\n✅ Best model for '{target}' saved to: {model_path}")
        print(f"✅ Scaler for '{target}' saved to: {scaler_path}")
            
    print("\n--- All models have been trained and saved successfully. ---")