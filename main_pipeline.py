# main_pipeline.py
import sqlite3
import requests
import time
from datetime import datetime
import concurrent.futures
import os

# --- Configuration ---
DATABASE_FILE_PATH = 'data/crypto_data.db'
COINS_TO_FETCH = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'ADA-USDT', 'XRP-USDT', 'DOGE-USDT']

def run_data_pipeline():
    """
    Fetches fresh data in parallel and creates/updates the raw data table.
    This is the first and most important step.
    """
    print("--- STARTING DATA PIPELINE: FETCHING AND STORING RAW DATA ---")
    
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(DATABASE_FILE_PATH), exist_ok=True)
    
    all_data_for_db = []
    
    def _fetch_single_coin_data(symbol):
        """Helper function to fetch data for one coin, designed to run in a thread."""
        days_to_fetch = 4000
        start_timestamp = int(time.time()) - (days_to_fetch * 24 * 60 * 60)
        url = f"https://api.kucoin.com/api/v1/market/candles?type=1day&symbol={symbol}&startAt={start_timestamp}"
        print(f"Fetching data for {symbol}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            raw_data = response.json()['data']
            processed_data = []
            for row in raw_data:
                processed_data.append((
                    int(row[0]) * 1000,
                    datetime.fromtimestamp(int(row[0])).strftime('%Y-%m-%d'),
                    symbol.replace('-', ''),
                    float(row[1]), float(row[3]), float(row[4]), float(row[2]),
                    float(row[5]), float(row[6])
                ))
            print(f"  > Successfully fetched {len(processed_data)} records for {symbol}.")
            return processed_data
        except requests.exceptions.RequestException as e:
            print(f"  > ERROR: Failed to fetch {symbol}. Reason: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(COINS_TO_FETCH)) as executor:
        for result in executor.map(_fetch_single_coin_data, COINS_TO_FETCH):
            all_data_for_db.extend(result)
            
    conn = sqlite3.connect(DATABASE_FILE_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS crypto_prices (id INTEGER PRIMARY KEY, Unix INTEGER, Date TEXT, Symbol TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume_Base REAL, Volume_Quote REAL, UNIQUE(Symbol, Unix))")
    cur.executemany("INSERT OR IGNORE INTO crypto_prices (Unix, Date, Symbol, Open, High, Low, Close, Volume_Base, Volume_Quote) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", all_data_for_db)
    conn.commit()
    print(f"\nDatabase is ready. Inserted {cur.rowcount} new records into 'crypto_prices'.")
    conn.close()
    print("\n--- DATA PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_data_pipeline()