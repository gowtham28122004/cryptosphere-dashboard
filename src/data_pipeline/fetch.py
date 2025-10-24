# src/data_pipeline/fetch.py
import requests
import sqlite3
import time
from datetime import datetime
import concurrent.futures

def run_data_pipeline(db_path, coins_to_fetch, start_date, end_date):
    """
    Fetches historical data for a specific date range and updates the database.
    This version is corrected to accept start_date and end_date.
    """
    all_data_for_db = []
    
    # Convert date objects to Unix timestamps in seconds
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(end_date.timetuple()))
    
    def _fetch_single_coin_data(symbol):
        # The KuCoin API uses start/end timestamps for date ranges
        url = f"https://api.kucoin.com/api/v1/market/candles?type=1day&symbol={symbol}&startAt={start_timestamp}&endAt={end_timestamp}"
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
            return processed_data
        except:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(coins_to_fetch)) as executor:
        for result in executor.map(_fetch_single_coin_data, coins_to_fetch):
            all_data_for_db.extend(result)
            
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS crypto_prices (id INTEGER PRIMARY KEY, Unix INTEGER, Date TEXT, Symbol TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume_Base REAL, Volume_Quote REAL, UNIQUE(Symbol, Unix))")
    cur.executemany("INSERT OR IGNORE INTO crypto_prices (Unix, Date, Symbol, Open, High, Low, Close, Volume_Base, Volume_Quote) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", all_data_for_db)
    conn.commit()
    conn.close()