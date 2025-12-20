"""
UI Module
Streamlit UI components and page layout logic
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import config
import services
import models
from styles import get_custom_css

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="StockSense Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def show_header():
    """Display main header"""
    st.markdown("""
    <div class="main-header fade-in">
        <div>
            <h1>StockSense Pro</h1>
            <p style="font-size: 1.2rem;">Advanced AI-Powered Stock Analysis Platform</p>
        </div>
        <div style="font-size: 4rem;">📈</div>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar():
    """Display sidebar with market data"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section fade-in">
            <h2 style="text-align: center; color: #00c8ff !important;">📊 Market Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Market Indices
        st.markdown("<div class='sidebar-section'><h4>📈 Live Indian Indices</h4>", unsafe_allow_html=True)
        indices = services.get_real_time_indices()
        for idx in indices:
            st.markdown(f"""
            <div style='margin: 0.8rem 0;'>
                <strong style='color: #00c8ff;'>{idx['name']}</strong><br>
                <span style='font-size: 1.1rem; font-weight: 600;'>{idx['price']}</span><br>
                <span style='color: {idx["color"]};'>{idx['change']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # FII/DII Activity
        fii_dii = services.get_fii_dii_data()
        st.markdown(f"""
        <div class='sidebar-section'>
            <h4>💰 FII/DII Activity</h4>
            <p style='color: #888; font-size: 0.9rem;'>Date: {fii_dii['date']}</p>
            <div style='margin: 1rem 0;'>
                <strong style='color: #00c8ff;'>FII</strong><br>
                <span style='color: #00ff88;'>Buy: ₹{fii_dii['FII']['buy']:.2f} Cr</span><br>
                <span style='color: #ff4444;'>Sell: ₹{fii_dii['FII']['sell']:.2f} Cr</span><br>
                <span style='color: {"#00ff88" if fii_dii["FII"]["net"] > 0 else "#ff4444"}; font-weight: 700;'>
                    Net: ₹{fii_dii['FII']['net']:+.2f} Cr
                </span>
            </div>
            <div style='margin: 1rem 0;'>
                <strong style='color: #00c8ff;'>DII</strong><br>
                <span style='color: #00ff88;'>Buy: ₹{fii_dii['DII']['buy']:.2f} Cr</span><br>
                <span style='color: #ff4444;'>Sell: ₹{fii_dii['DII']['sell']:.2f} Cr</span><br>
                <span style='color: {"#00ff88" if fii_dii["DII"]["net"] > 0 else "#ff4444"}; font-weight: 700;'>
                    Net: ₹{fii_dii['DII']['net']:+.2f} Cr
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Latest News
        st.markdown("<div class='sidebar-section'><h4>📰 Latest Market News</h4>", unsafe_allow_html=True)
        for news in services.get_latest_indian_market_news():
            st.markdown(f"<p style='margin: 0.5rem 0; font-size: 0.9rem;'>{news}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Market Wisdom
        st.markdown("""
        <div class='sidebar-section'>
            <h4>💡 Market Wisdom</h4>
            <blockquote style='border-left: 3px solid #00c8ff; padding-left: 1rem; font-style: italic; color: #ccc;'>
                "In investing, what is comfortable is rarely profitable."
                <br><span style='color: #00c8ff;'>— Robert Arnott</span>
            </blockquote>
        </div>
        """, unsafe_allow_html=True)

def show_trending_stocks():
    """Display trending stocks section"""
    st.markdown("## 🔥 Trending Indian Stocks Today")
    trending_data = services.get_trending_stocks()
    cols = st.columns(3)
    for i, stock in enumerate(trending_data):
        with cols[i % 3]:
            change_color = "#00ff88" if stock['change'] > 0 else "#ff4444"
            st.markdown(f"""
            <div class="trending-card">
                <h4 style="margin: 0; color: #00c8ff;">{stock['symbol']}</h4>
                <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #aaa;">{stock['company']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: 600;">₹{stock['price']:.2f}</p>
                <p style="margin: 0; color: {change_color}; font-weight: 600;">{stock['change']:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

def show_stock_selector():
    """Display stock selection and analysis button"""
    st.markdown("""
    <div class="search-container fade-in">
        <h2 style="font-size: 2rem; margin-bottom: 1rem;">🎯 Stock Analysis</h2>
        <p style="font-size: 1.1rem;">Select any Nifty 50 stock for comprehensive Buy/Hold/Sell recommendation powered by LSTM & Random Forest AI</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_stock = st.selectbox(
            "Choose a Nifty 50 Stock:",
            config.NIFTY_50_STOCKS,
            format_func=lambda x: f"{x.replace('.NS', '')} - {config.COMPANY_NAMES.get(x, x)}"
        )
        
        if st.button("🚀 Analyze Stock", type="primary", use_container_width=True):
            analyze_selected_stock(selected_stock)
    
    return selected_stock

def analyze_selected_stock(ticker):
    """Analyze selected stock and show results"""
    with st.spinner("🔄 Analyzing stock. This may take a moment..."):
        # Fetch data
        df = services.fetch_stock_data(ticker)
        if df is None or df.empty:
            st.error("Could not fetch data for this stock. Please try again.")
            return
        
        # Calculate indicators
        df = models.calculate_technical_indicators(df)
        
        # Get news
        company_name = config.COMPANY_NAMES.get(ticker.upper(), ticker.replace('.NS', ''))
        news = services.fetch_news_newsapi(company_name, limit=3)
        
        # Train models
        model, scaler = models.train_lstm(df)
        labels = models.create_labels(df)
        features = models.build_features(df)
        clf, X_train, X_test, y_train, y_test = models.train_rf(features, labels)
        
        if clf is None:
            st.error("Could not train model for this stock.")
            return
        
        # Get recommendation
        recommendation, confidence, details = models.get_stock_recommendation(ticker, df, model, scaler, clf, features)
        
        # Store in session state
        st.session_state.ticker = ticker
        st.session_state.df = df
        st.session_state.news = news
        st.session_state.recommendation = recommendation
        st.session_state.confidence = confidence
        st.session_state.details = details
        st.session_state.clf = clf
        st.session_state.labels = labels
        st.session_state.features = features
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.model = model
        st.session_state.scaler = scaler
        
        st.rerun()

def show_recommendation_result():
    """Display stock recommendation result"""
    if 'recommendation' not in st.session_state:
        return
    
    recommendation = st.session_state.recommendation
    confidence = st.session_state.confidence
    details = st.session_state.details
    ticker = st.session_state.ticker
    
    # Determine color and style
    if recommendation == "BUY":
        result_class = "buy-result"
        emoji = "🟢"
    elif recommendation == "SELL":
        result_class = "sell-result"
        emoji = "🔴"
    else:
        result_class = "hold-result"
        emoji = "🟡"
    
    st.markdown(f"""
    <div class="recommendation-result {result_class}">
        <h2 style="margin: 0;">{emoji} {recommendation}</h2>
        <h3 style="margin: 1rem 0; font-size: 1.8rem;">{ticker.replace('.NS', '')}</h3>
        <p style="margin: 1rem 0; font-size: 1.1rem;">{details}</p>
        <h4 style="margin-top: 1.5rem; margin-bottom: 0.5rem;">Confidence Score</h4>
        <p style="margin: 0; font-size: 1.3rem; font-weight: 700;">{confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

def show_technical_charts():
    """Display technical indicator charts"""
    if 'df' not in st.session_state:
        return
    
    df = st.session_state.df
    ticker = st.session_state.ticker
    
    st.markdown("### 📊 Technical Analysis Charts")
    
    # Price chart with indicators
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                              line=dict(color='#00c8ff', width=2)))
    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', 
                              line=dict(color='orange', width=1.5, dash='dash')))
    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA 200', 
                              line=dict(color='red', width=1.5, dash='dash')))
    fig1.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA 50', 
                              line=dict(color='yellow', width=1, dash='dot')))
    fig1.update_layout(
        title=f'{ticker.replace(".NS", "")} - Price & Moving Averages',
        height=450,
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Price (₹)'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # RSI and MACD
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                                  line=dict(color='purple', width=2)))
        fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig2.add_hline(y=50, line_dash="dot", line_color="gray")
        fig2.update_layout(
            title='RSI Indicator',
            height=350,
            template='plotly_dark',
            yaxis_title='RSI Value'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = go.Figure()
        colors = ['green' if val > 0 else 'red' for val in df['MACD']]
        fig3.add_trace(go.Bar(x=df.index, y=df['MACD'], name='MACD', 
                              marker_color=colors))
        fig3.update_layout(
            title='MACD Histogram',
            height=350,
            template='plotly_dark',
            yaxis_title='MACD Value'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Bollinger Bands
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band',
                              line=dict(color='red', width=1, dash='dash')))
    fig4.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price',
                              line=dict(color='#00c8ff', width=2)))
    fig4.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band',
                              line=dict(color='green', width=1, dash='dash')))
    fig4.update_layout(
        title='Bollinger Bands',
        height=400,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price (₹)'
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Volume chart
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                          marker_color='rgba(0, 200, 255, 0.5)'))
    fig5.update_layout(
        title='Trading Volume',
        height=300,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Volume'
    )
    st.plotly_chart(fig5, use_container_width=True)

def show_model_performance():
    """Display model performance metrics"""
    if 'clf' not in st.session_state or st.session_state.clf is None:
        return
    
    clf = st.session_state.clf
    X_test = st.session_state.X_test
    
    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        st.markdown("#### 🔍 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance in Prediction Model',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_news_section():
    """Display relevant news"""
    if 'news' not in st.session_state or not st.session_state.news:
        return
    
    st.markdown("### 📰 Relevant News")
    for article in st.session_state.news:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{article.get('title', 'N/A')[:100]}...**")
        with col2:
            st.markdown(f"[Read →]({article.get('url', '#')})")

def show_backtest_section():
    """Display backtest analysis"""
    if st.button("📊 Run Portfolio Backtest", type="secondary", use_container_width=False):
        if 'df' not in st.session_state:
            st.error("Please analyze a stock first.")
            return
        
        ticker = st.session_state.ticker
        df = st.session_state.df
        clf = st.session_state.clf
        labels = st.session_state.labels
        features = st.session_state.features
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        
        st.markdown(f"### 📈 Backtesting Analysis for {ticker.replace('.NS', '')}")
        
        with st.spinner("Running backtest simulation..."):
            # Generate signals
            signals = pd.Series(clf.predict(features.fillna(0)), index=features.index)
            portfolio_df, trades, trade_returns, summary = models.portfolio_backtest(
                df, dict(zip(df.index, signals))
            )
            
            if summary:
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Cumulative Return", f"{summary['cumulative_return']:.2f}%")
                
                with col2:
                    st.metric("📈 Annualized Return", f"{summary['annualized_return']:.2f}%")
                
                with col3:
                    st.metric("📉 Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
                
                with col4:
                    st.metric("🔄 Total Trades", summary['num_trades'])
                
                # Portfolio value over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['PortfolioValue'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00c8ff', width=2)
                ))
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_dark',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for backtesting.")

def show_sectoral_performance():
    """Display sectoral performance overview"""
    st.markdown("---")
    st.markdown("## 📊 Sectoral Performance Overview")
    sectors, performance = services.get_sectoral_performance()
    
    fig = go.Figure()
    colors = ['#00C851' if p > 0 else '#ff4444' for p in performance]
    fig.add_trace(go.Bar(
        x=sectors,
        y=performance,
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in performance],
        textposition='outside'
    ))
    fig.update_layout(
        title='Sectoral Performance Today',
        template='plotly_dark',
        height=400,
        yaxis_title='Change (%)',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def show_footer():
    """Display footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #888;'>
        <p><strong>StockSense Pro</strong> - Advanced Stock Analysis Platform</p>
        <p style='font-size: 0.9rem;'>Powered by LSTM with Attention Mechanism & Random Forest Classifier</p>
    </div>
    """, unsafe_allow_html=True)
