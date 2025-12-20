"""
Services Module
Business logic for data fetching, news, and market operations
"""
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import streamlit as st
import config

@st.cache_data(ttl=config.CACHE_TTL_INDICES)
def get_real_time_indices():
    """Get real-time index data"""
    indices_data = []
    for name, symbol in config.INDIAN_INDICES.items():
        try:
            data = yf.Ticker(symbol).history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Open'].iloc[-1]
                change_pct = ((current_price - prev_close) / prev_close * 100)
                indices_data.append({
                    'name': name,
                    'price': f"{current_price:,.2f}",
                    'change': f"{change_pct:+.2f}%",
                    'color': '#00ff88' if change_pct > 0 else '#ff4444'
                })
        except:
            pass
    return indices_data

@st.cache_data(ttl=config.CACHE_TTL_TRENDING)
def get_trending_stocks():
    """Get trending stocks data"""
    trending_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS']
    trending_data = []
    for ticker in trending_stocks:
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Open'].iloc[-1]
                change_pct = ((current_price - prev_close) / prev_close * 100)
                trending_data.append({
                    "symbol": ticker.replace('.NS', ''),
                    "company": config.COMPANY_NAMES.get(ticker, ticker),
                    "price": current_price,
                    "change": change_pct
                })
        except:
            pass
    return trending_data[:6] if trending_data else config.DEFAULT_TRENDING_STOCKS

@st.cache_data(ttl=config.CACHE_TTL_NEWS)
def get_latest_indian_market_news():
    """Fetch latest market news from NewsAPI"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'Indian stock market OR Nifty OR Sensex',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'apiKey': config.NEWSAPI_KEY
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if articles:
                return [f"📰 {a.get('title', '')[:80]}" for a in articles[:5]]
    except:
        pass
    return config.DEFAULT_NEWS

def get_fii_dii_data():
    """Get FII/DII data"""
    return {
        "FII": config.DEFAULT_FII_DII["FII"],
        "DII": config.DEFAULT_FII_DII["DII"],
        "date": datetime.now().strftime("%Y-%m-%d")
    }

@st.cache_data(ttl=config.CACHE_TTL_NEWS)
def get_sectoral_performance():
    """Get sectoral performance data"""
    return config.SECTORS, config.SECTOR_PERFORMANCE

def fetch_news_newsapi(query, limit=5):
    """Fetch news from NewsAPI"""
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={config.NEWSAPI_KEY}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])
            return [{"title": a.get("title", ""), "url": a.get("url", "")} for a in articles[:limit]]
    except:
        pass
    return []

def fetch_stock_data(ticker, period=None):
    """Fetch stock data from yfinance"""
    if period is None:
        period = config.DATA_PERIOD
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty or len(df) < config.MIN_DATA_POINTS:
            return None
        return df.rename(columns=str.capitalize)
    except:
        return None
