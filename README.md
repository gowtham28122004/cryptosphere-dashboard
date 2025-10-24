# CryptoSphere: End-to-End Crypto Analysis & Prediction Dashboard

This project is a complete, end-to-end data science application built in Python. It automates the entire workflow of a financial analysis project, from parallel data ingestion and robust database storage to advanced portfolio analysis, real-time risk monitoring, and predictive modeling with a deep learning LSTM network. The entire system is controlled through a user-friendly, interactive web dashboard built with Streamlit, complete with secure user authentication.

## 🚀 Key Features

This project demonstrates a full data science and engineering lifecycle, including:

-   **Parallel Data Ingestion:** Fetches historical daily data for 6 major cryptocurrencies (BTC, ETH, SOL, ADA, XRP, DOGE) simultaneously from a free public API.
-   **Robust Data Storage:** Stores all raw historical data and analysis results in a local SQLite database (`crypto_data.db`), using `UNIQUE` constraints to ensure data integrity.
-   **Advanced Portfolio Analysis:**
    -   Implements and stress-tests **9 distinct portfolio weighting strategies**, including static rules (e.g., `Risk Level`, `Market Cap`) and dynamic, data-driven rules (`Risk-Parity`, `Sharpe-Maximization`, `Momentum`).
    -   Enforces a **40% maximum weight** on any single asset to ensure diversification.
-   **Database for Metrics:** Stores all calculated analysis results (portfolio returns, risk, and asset weights) in dedicated `portfolio` and `portfolio_assets` tables.
-   **Real-Time Risk Monitoring:**
    -   Implements **6 key risk rules** (Volatility, Sharpe Ratio, Max Drawdown, Sortino Ratio, Beta, and Asset Concentration).
    -   Allows users to run checks on any portfolio or coin and stores the PASS/FAIL results in an `risk_assessment` table.
-   **Automated & Personalized Alerting:** Automatically sends a detailed **email alert to the logged-in user's email address** if any of the monitored risk rules fail.
-   **Predictive Modeling (LSTM):**
    -   A "Model Factory" script trains and saves an optimized **LSTM model** for every individual coin and every portfolio strategy, using `TimeSeriesSplit` cross-validation for robustness.
    -   The final models achieve high accuracy (e.g., **R-squared of 0.88** and **MAPE of 3.00%** for Bitcoin) in tracking the price trend.
-   **Secure User Authentication:** A complete Login/Signup system using email and hashed passwords, with a separate `users.db` for secure credential storage.
-   **Interactive Dashboard (Streamlit):** A user-friendly, multi-page web application that serves as the central control panel for the entire project.

## 🛠️ Technologies Used

-   **Programming Language:** Python
-   **Core Libraries:**
    -   `requests`: For API calls.
    -   `sqlite3`: For database interaction.
    -   `pandas` & `numpy`: For data manipulation and analysis.
    -   `concurrent.futures`: For parallel processing.
    -   `matplotlib`: For data visualization.
    -   `scikit-learn`: For data scaling, model evaluation, and cross-validation (`GridSearchCV`).
    -   `tensorflow` / `keras`: For building and training the LSTM models.
    -   `scikeras`: To wrap Keras models for use with scikit-learn.
    -   `smtplib`: For the email alerting system.
-   **Web Dashboard:** `streamlit`
-   **Data Storage:** SQLite

## 🏃 How to Run the Project

This project is designed to be run locally.

### Prerequisites
1.  Python 3.8+
2.  Git
3.  For email alerts, a **Gmail account** with **2-Step Verification** enabled and a **16-digit App Password**.

### Installation
1.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
1.  **Create a secrets file:** In the main project folder, create a new folder named `.streamlit`.
2.  Inside `.streamlit/`, create a new file named `secrets.toml`.
3.  Add your email credentials to this file:
    ```toml
    # .streamlit/secrets.toml
    SENDER_EMAIL = "your_email@gmail.com"
    SENDER_PASSWORD = "abcd efgh ijkl mnop" # Your 16-digit Google App Password
    ```

### Execution Steps
The project requires a two-step execution process: first build the data and models, then run the dashboard.

**Step 1: Run the Data Pipeline**
This script will fetch all historical data and create the `crypto_data.db` file.
```bash
python data_pipeline.py
