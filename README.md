
# StockSense Pro - Modular Architecture

## Project Structure



```
stock/
├── app.py              # Main entry point (orchestrator)
├── config.py           # Configuration & Constants
├── services.py         # Business Logic (data fetching, news, market data)
├── models.py           # ML & Deep Learning Models (LSTM, Random Forest)
├── styles.py           # CSS Styling
├── ui.py               # Streamlit UI Components
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
└── venv/               # Virtual environment
```

---

##  Module Descriptions

### **1. `app.py` - Main Entry Point**
- Orchestrates all modules
- Minimal business logic
- Clean and readable main flow
- All functionality delegated to other modules

### **2. `config.py` - Configuration & Constants**
- **API Keys**: Loaded from `.env` file
- **Model Parameters**: 
  - `LOOKBACK`, `EPOCHS`, `HIDDEN_SIZE`, `BATCH_SIZE`
- **ML Parameters**: 
  - `RF_N_ESTIMATORS`, `RF_MAX_DEPTH`, `RF_TEST_SIZE`
- **Data Constants**: Stock lists, company names, indices
- **Cache TTL**: API cache timeouts
- **Default Fallback Data**: For when APIs fail

### **3. `services.py` - 
Functions for:
- `get_real_time_indices()` - Fetch market indices
- `get_trending_stocks()` - Get trending stocks data
- `get_latest_indian_market_news()` - Fetch market news
- `get_fii_dii_data()` - Get FII/DII activity
- `get_sectoral_performance()` - Sectoral analysis
- `fetch_news_newsapi()` - News fetching
- `fetch_stock_data()` - Download OHLCV data

All with caching to optimize performance.

### **4. `models.py` - ML & Deep Learning**
**LSTM with Attention:**
- `AttentionLayer` - Custom attention mechanism
- `build_lstm_model()` - LSTM architecture
- `train_lstm()` - Model training
- `lstm_predict()` - Price prediction

**Technical Indicators:**
- `calculate_technical_indicators()` - SMA, EMA, RSI, MACD, Bollinger Bands
- `interpret_signals()` - Technical signal interpretation

**Random Forest Classifier:**
- `create_labels()` - BUY/HOLD/SELL labels
- `build_features()` - Feature engineering
- `train_rf()` - Random Forest training

**Backtesting:**
- `portfolio_backtest()` - Simulate trading strategy
- Performance metrics (returns, Sharpe ratio, etc.)

### **5. `styles.py` - CSS Styling**
- `get_custom_css()` - Returns all CSS as a single function
- Gradient backgrounds, animations, responsive design
- Card styling, button styling, theme colors
- Can be easily modified for different themes

### **6. `ui.py` - Streamlit Components**
**Page Setup:**
- `setup_page_config()` - Streamlit configuration
- `apply_custom_css()` - Apply styling

**Layout Components:**
- `show_header()` - Main header
- `show_sidebar()` - Dashboard sidebar
- `show_trending_stocks()` - Trending stocks cards
- `show_stock_selector()` - Stock selection dropdown

**Analysis Components:**
- `show_recommendation_result()` - BUY/SELL/HOLD result
- `show_technical_charts()` - Technical analysis plots
- `show_model_performance()` - Feature importance
- `show_news_section()` - Related news
- `show_backtest_section()` - Backtest analysis
- `show_sectoral_performance()` - Sector analysis
- `show_footer()` - Footer

### **7. `.env` - Environment Variables**
Store sensitive data:
```env
NEWSAPI_KEY=your_api_key_here
LOOKBACK=60
EPOCHS=20
HIDDEN_SIZE=64
...
```

---

##  How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update `.env` with your API keys** (if needed)

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```


  


##  Architecture Diagram

```
┌─────────────────────────────────────────────┐
│              app.py (Main)                  │
│         Orchestrates all modules            │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
        ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │config. │ │services│ │models. │ │ styles │
    │  py    │ │  py    │ │  py    │ │  py    │
    └────────┘ └────────┘ └────────┘ └────────┘
        │          │          │          │
        └──────────┴──────────┴──────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │     ui.py            │
        │  (All UI components) │
        └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Streamlit App      │
        │  (User Interface)    │
        └──────────────────────┘
```

---

