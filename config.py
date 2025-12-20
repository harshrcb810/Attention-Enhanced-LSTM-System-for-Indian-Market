"""
Configuration and Constants Module
Loads all settings from .env and centralizes configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============ API KEYS ============
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "272cce001c674f2b8fe9bb051b2c1804")

# ============ MODEL PARAMETERS ============
LOOKBACK = int(os.getenv("LOOKBACK", "60"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "64"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.1"))

# ============ ML MODEL PARAMETERS ============
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "100"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "10"))
RF_TEST_SIZE = float(os.getenv("RF_TEST_SIZE", "0.2"))
TRADING_HORIZON = int(os.getenv("TRADING_HORIZON", "3"))

# ============ DATA FETCH PARAMETERS ============
DATA_PERIOD = os.getenv("DATA_PERIOD", "2y")
MIN_DATA_POINTS = int(os.getenv("MIN_DATA_POINTS", "200"))
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "5"))
CACHE_TTL_INDICES = int(os.getenv("CACHE_TTL_INDICES", "300"))
CACHE_TTL_TRENDING = int(os.getenv("CACHE_TTL_TRENDING", "300"))
CACHE_TTL_NEWS = int(os.getenv("CACHE_TTL_NEWS", "1800"))

# ============ NIFTY 50 STOCKS ============
NIFTY_50_STOCKS = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
    'LT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'TITAN.NS',
    'NESTLEIND.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'HCLTECH.NS', 'BAJFINANCE.NS',
    'TECHM.NS', 'SUNPHARMA.NS', 'POWERGRID.NS', 'NTPC.NS', 'COALINDIA.NS'
]

COMPANY_NAMES = {
    'RELIANCE.NS': 'Reliance Industries', 'HDFCBANK.NS': 'HDFC Bank', 'TCS.NS': 'TCS',
    'INFY.NS': 'Infosys', 'HINDUNILVR.NS': 'Hindustan Unilever', 'ICICIBANK.NS': 'ICICI Bank',
    'KOTAKBANK.NS': 'Kotak Bank', 'BHARTIARTL.NS': 'Bharti Airtel', 'ITC.NS': 'ITC Ltd',
    'SBIN.NS': 'SBI', 'LT.NS': 'L&T', 'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank', 'MARUTI.NS': 'Maruti Suzuki', 'TITAN.NS': 'Titan',
    'NESTLEIND.NS': 'Nestle India', 'WIPRO.NS': 'Wipro', 'ULTRACEMCO.NS': 'UltraTech',
    'HCLTECH.NS': 'HCL Tech', 'BAJFINANCE.NS': 'Bajaj Finance', 'TECHM.NS': 'Tech Mahindra',
    'SUNPHARMA.NS': 'Sun Pharma', 'POWERGRID.NS': 'Power Grid', 'NTPC.NS': 'NTPC',
    'COALINDIA.NS': 'Coal India'
}

INDIAN_INDICES = {
    'Nifty 50': '^NSEI',
    'Sensex': '^BSESN',
    'Nifty Bank': '^NSEBANK',
    'Nifty IT': '^CNXIT'
}

# ============ SECTORAL DATA ============
SECTORS = ['Banking', 'IT Services', 'Oil & Gas', 'Consumer Goods', 'Automobiles', 'Pharma', 'Metals', 'Telecom']
SECTOR_PERFORMANCE = [2.1, 1.5, -0.8, 1.9, 2.8, -0.3, -1.5, 0.7]

# ============ DEFAULT FII/DII DATA ============
DEFAULT_FII_DII = {
    "FII": {"buy": 7500.25, "sell": 6200.50, "net": 1299.75},
    "DII": {"buy": 4800.75, "sell": 5100.25, "net": -299.50},
}

DEFAULT_NEWS = [
    "🔥 Nifty 50 hits fresh record high",
    "💰 FII inflows boost market sentiment",
    "🏦 Banking stocks surge on rate cut hopes",
    "💻 IT sector shows resilience",
    "🚗 Auto stocks rally on festive demand"
]

DEFAULT_TRENDING_STOCKS = [
    {"symbol": "RELIANCE", "company": "Reliance Industries", "price": 2456.30, "change": 2.3},
    {"symbol": "TCS", "company": "Tata Consultancy Services", "price": 3789.45, "change": 1.8},
]
