import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta
import os

# Import all custom functions
from src.data_pipeline.fetch import run_data_pipeline
from src.analysis.portfolio_math import assign_weights, calculate_portfolio_return
from src.analysis.risk_checker import run_risk_check_logic, send_email_alert
from src.prediction.model import get_historical_performance, predict_future_price, load_model_and_scaler
from src.auth import create_users_table, add_user, check_user

# --- Page Configuration ---
st.set_page_config(page_title="CryptoSphere Dashboard", page_icon="🌐", layout="wide")

# --- Initialize User Database ---
create_users_table()

# --- Global Config & Session State ---
DATABASE_FILE_PATH = 'data/crypto_data.db'
MODELS_DIR = 'models/'
REAL_COINS_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT']
COINS_TO_FETCH = REAL_COINS_SYMBOLS  # FIX: Added missing variable
PORTFOLIO_RULES = ['risk_level', 'market_cap', 'safety', 'growth', 'equal', 'risk_parity', 'sharpe_max', 'momentum']

if 'authenticated' not in st.session_state:
    st.session_state.update({'authenticated': False, 'username': '', 'email': '', 'page': '🏠 Overview'})

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_all_prices_from_db():
    try:
        conn = sqlite3.connect(DATABASE_FILE_PATH, check_same_thread=False)
        df = pd.read_sql_query("SELECT Date, Symbol, Close FROM crypto_prices ORDER BY Date ASC", conn, parse_dates=['Date'])
        conn.close(); return df.pivot(index='Date', columns='Symbol', values='Close')
    except: return pd.DataFrame()

# ==============================================================================
# PAGE RENDERING FUNCTIONS (Complete and Not Cut Off)
# ==============================================================================
def render_overview_page():
    st.markdown("""<style>.welcome-banner{padding:2.5rem 2rem;background-image:linear-gradient(to right, rgba(0,40,80,0.9),rgba(0,70,120,0.7)),url('https://images.unsplash.com/photo-1621418359569-41a7737e6c9a');background-size:cover;border-radius:10px;color:white;margin-bottom:2rem}.welcome-banner h1{color:white}</style>""", unsafe_allow_html=True)
    st.markdown(f'<div class="welcome-banner"><h1>Welcome, {st.session_state["username"].capitalize()}!</h1><p>Your personal dashboard for the end-to-end crypto analysis project.</p></div>', unsafe_allow_html=True)
    st.header("Project Features:"); st.markdown("- **Dynamic Data Pipeline**\n- **Portfolio Strategy Explorer**\n- **Real-Time Risk Monitor**\n- **Predictive Modeling**")

def render_pipeline_page():
    st.title("🔄 Data Pipeline & Viewer")
    st.markdown("Fetch historical data for a specific date range from the API or view the data currently in the database.")
    with st.expander("Fetch New Data from API"):
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = c2.date_input("End Date", datetime.now())
        if st.button("🚀 Fetch Data for Date Range"):
            if start_date >= end_date: st.error("Error: Start date must be before end date.")
            else:
                with st.spinner("Fetching data in parallel..."):
                    run_data_pipeline(DATABASE_FILE_PATH, COINS_TO_FETCH, start_date, end_date)
                st.success("Pipeline complete! The database has been updated.")
                st.cache_data.clear()
                st.rerun()
    st.header("Data Explorer")
    prices_df = load_all_prices_from_db()
    if not prices_df.empty:
        start_v, end_v = st.select_slider("Select a date range to view:", options=prices_df.index, value=(prices_df.index.max() - timedelta(days=90), prices_df.index.max()))
        st.dataframe(prices_df.loc[start_v:end_v])
    else:
        st.warning("No data found. Please run the data pipeline to populate the database.")

def render_portfolio_page():
    st.title("📊 Portfolio Strategy Explorer")
    prices_df = load_all_prices_from_db()
    if prices_df.empty:
        st.warning("No data found. Please run the data pipeline first.")
        return
    returns_df = prices_df.pct_change()
    rule = st.selectbox("Select a Portfolio Strategy", PORTFOLIO_RULES)
    weights = assign_weights(rule, REAL_COINS_SYMBOLS, daily_returns_df=returns_df)

    st.header("Performance Metrics")
    portfolio_returns = calculate_portfolio_return(weights, returns_df).dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Daily Return", f"{portfolio_returns.mean() * 100:.4f}%")
    c2.metric("Volatility (Risk)", f"{portfolio_returns.std() * 100:.4f}%")
    sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(365) if portfolio_returns.std() != 0 else 0
    c3.metric("Annualized Sharpe Ratio", f"{sharpe:.4f}")
    
    col_chart, col_weights = st.columns(2)
    with col_weights:
        st.header(f"Asset Weights")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', textprops={'fontsize': 8})
        ax.axis('equal')
        st.pyplot(fig)
    with col_chart:
        st.header("Growth of $100")
        growth = 100 * (1 + portfolio_returns).cumprod()
        st.line_chart(growth)

def render_risk_page():
    st.title("🚨 Interactive Risk Monitor")
    st.markdown("Select any portfolio strategy or individual coin to run a real-time risk assessment.")
    prices_df = load_all_prices_from_db()
    if prices_df.empty:
        st.warning("No data found. Please run the data pipeline first.")
        return
    returns_df = prices_df.pct_change()
    target = st.selectbox("Select Portfolio or Coin to Check", PORTFOLIO_RULES + REAL_COINS_SYMBOLS)
    if st.button(f"Run Risk Check for {target}"):
        with st.spinner("Calculating risk..."):
            st.session_state.risk_results_df = run_risk_check_logic(target, returns_df * 100, REAL_COINS_SYMBOLS, PORTFOLIO_RULES)
    if 'risk_results_df' in st.session_state and st.session_state.risk_results_df is not None:
        risk_df = st.session_state.risk_results_df
        risk_df.index = np.arange(1, len(risk_df) + 1)
        def color_status(val):
            return 'color: red' if 'FAIL' in val else ('color: grey' if val == 'N/A' else 'color: green')
        st.header("Risk Assessment Summary")
        st.dataframe(risk_df.style.apply(lambda x: [color_status(v) for v in x], subset=['Status']))
        failed = risk_df[risk_df['Status'].str.contains('FAIL')]
        if not failed.empty:
            st.warning("⚠️ One or more risk rules have failed!")
            if st.button("📧 Send Alert to My Email"):
                try:
                    s, p, r = st.secrets["SENDER_EMAIL"], st.secrets["SENDER_PASSWORD"], st.session_state['email']
                    with st.spinner(f"Sending alert..."):
                        send_email_alert(failed, s, p, r)
                    st.success("Email alert sent!")
                except Exception as e:
                    st.error(f"Failed to send email. Error: {e}")

def render_prediction_page():
    st.title("🔮 Interactive Price Prediction (LSTM)")
    st.markdown("Select a pre-trained model to view its historical performance or make a new prediction.")
    prices_df = load_all_prices_from_db()
    if not os.path.isdir(MODELS_DIR) or not os.listdir(MODELS_DIR):
        st.error("No trained models found. Please run the `train_models.py` script first.")
    else:
        models = sorted([f.replace('_model.keras', '') for f in os.listdir(MODELS_DIR) if f.endswith('.keras')])
        target = st.selectbox("Select a Model", models)
        if target and not prices_df.empty:
            m_path = os.path.join(MODELS_DIR, f"{target}_model.keras")
            s_path = os.path.join(MODELS_DIR, f"{target}_scaler.joblib")
            model, scaler = load_model_and_scaler(m_path, s_path)
            returns_df = prices_df.pct_change()
            if target in REAL_COINS_SYMBOLS:
                series = prices_df[target]
            else:
                weights = assign_weights(target, REAL_COINS_SYMBOLS, daily_returns_df=returns_df)
                p_returns = calculate_portfolio_return(weights, returns_df)
                series = 100 * (1 + p_returns).cumprod()
            series.dropna(inplace=True)
            
            st.header("Live Prediction Tool")
            date = st.date_input("Select a Date to Predict", datetime.now() + timedelta(days=1))
            if st.button("🚀 Predict Price"):
                with st.spinner("Making prediction..."):
                    res = predict_future_price(model, scaler, series, date)
                    if 'error' in res:
                        st.error(res['error'])
                    else:
                        st.metric(label=f"Predicted Price for {target}", value=f"${res['predicted_price']:,.2f}", delta=f"{((res['predicted_price'] / res['last_known_price']) - 1):.2%} vs. previous day")
            
            with st.expander("View Model's Historical Performance"):
                metrics, plot_df = get_historical_performance(model, scaler, series)
                if 'error' in metrics:
                    st.error(metrics['error'])
                else:
                    st.subheader(f"Evaluation Metrics")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("RMSE", f"${metrics['rmse']:,.2f}")
                    c2.metric("MAE", f"${metrics['mae']:,.2f}")
                    c3.metric("MAPE", f"${metrics['mape']:.2f}%")
                    c4.metric("R²", f"${metrics['r2']:.4f}")
                    st.subheader("Historical Actual vs. Predicted Prices")
                    st.line_chart(plot_df)

# --- Authentication Page ---
def login_page():
    st.title("Welcome to CryptoSphere"); st.info("Please login or sign up.")
    login_tab, signup_tab = st.tabs(["🔐 Login", "✍️ Sign Up"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email").lower(); password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                username = check_user(email, password)
                if username: st.session_state.update({'authenticated': True, 'username': username, 'email': email}); st.rerun()
                else: st.error("Invalid email or password.")
    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Display Name"); email = st.text_input("Email").lower(); pword = st.text_input("Password", type="password")
            if st.form_submit_button("Sign Up"):
                if not all([name, pword, email]): st.warning("All fields are required.")
                elif add_user(name, pword, email): st.success("Account created! Please Login.")
                else: st.error("Email already exists.")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if not st.session_state.get('authenticated', False):
    login_page()
else:
    # --- Fixed Navigation Header ---
    st.markdown("""<style> div.block-container {padding-top: 3.5rem;} </style>""", unsafe_allow_html=True)
    header_cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 3, 1.5], vertical_alignment="center")
    
    # FIX: Made page names consistent between navigation and routing
    nav_items = {
        "🏠 Overview": "🏠 Overview",
        "🔄 Pipeline": "🔄 Pipeline",
        "📊 Analysis": "📊 Analysis",
        "🚨 Risk": "🚨 Risk",
        "🔮 Predict": "🔮 Predict"
    }
    
    for i, (label, page_name) in enumerate(nav_items.items()):
        if header_cols[i].button(label, use_container_width=True, key=f"nav_{i}"):
            st.session_state.page = page_name
            st.rerun()
    
    with header_cols[5]:
        st.markdown(f"<p style='text-align: right; color: grey; font-size: 0.9em; margin: 0;'>Logged in as: {st.session_state['username']}</p>", unsafe_allow_html=True)
    
    with header_cols[6]:
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    
    # FIX: Updated page routing to match navigation
    page_name = st.session_state.get('page', '🏠 Overview')
    if page_name == "🏠 Overview":
        render_overview_page()
    elif page_name == "🔄 Pipeline":
        render_pipeline_page()
    elif page_name == "📊 Analysis":
        render_portfolio_page()
    elif page_name == "🚨 Risk":
        render_risk_page()
    elif page_name == "🔮 Predict":
        render_prediction_page()