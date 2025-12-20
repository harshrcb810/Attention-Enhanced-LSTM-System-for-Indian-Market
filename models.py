"""
Models Module
LSTM with Attention mechanism, Random Forest classifier, and technical indicators
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import config

# ============ ATTENTION LAYER ============
class AttentionLayer(layers.Layer):
    """Custom Attention Layer for LSTM"""
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # inputs shape: (batch, time_steps, features)
        # har time-step ka importance score
        scores = tf.reduce_mean(inputs, axis=2)
        # scores ko weights me convert (0–1, sum = 1)
        weights = tf.nn.softmax(scores, axis=1)
        # important time-steps ka weighted sum
        context = tf.reduce_sum(
            inputs * tf.expand_dims(weights, -1),
            axis=1
        )
        return context


# ============ MODEL BUILDING ============
def build_lstm_model(input_shape, hidden_size):
    """Build LSTM model with Attention mechanism"""
    inputs = keras.Input(shape=input_shape)
    
    # LSTM learns sequence patterns
    lstm_out = layers.LSTM(
        hidden_size,
        return_sequences=True
    )(inputs)
    
    # Attention selects important time-steps
    attention_out = AttentionLayer()(lstm_out)
    
    # Final price prediction
    output = layers.Dense(1)(attention_out)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model


# ============ TECHNICAL INDICATORS ============
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df["SMA50"] = df["Close"].rolling(50, min_periods=10).mean()
    df["SMA200"] = df["Close"].rolling(200, min_periods=50).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14).mean() / down.replace(0, 1).ewm(alpha=1/14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["MACD"] = macd - signal
    
    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    
    # Rate of Change
    df["roc_5"] = df["Close"].pct_change(5)
    df["roc_10"] = df["Close"].pct_change(10)
    
    return df.fillna(method='bfill').fillna(0)


def interpret_signals(df):
    """Interpret technical signals"""
    sig = {}
    sig["SMA"] = "Bullish" if df["SMA50"].iloc[-1] > df["SMA200"].iloc[-1] else "Bearish"
    sig["EMA"] = "Bullish" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "Bearish"
    rsi = df["RSI"].iloc[-1]
    sig["RSI"] = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
    sig["MACD"] = "Bullish" if df["MACD"].iloc[-1] > 0 else "Bearish"
    close = df["Close"].iloc[-1]
    sig["Bollinger"] = "Near Lower" if close <= df["BB_lower"].iloc[-1] else "Near Upper" if close >= df["BB_upper"].iloc[-1] else "Neutral"
    sig["ROC5"] = "Up" if df["roc_5"].iloc[-1] > 0 else "Down"
    return sig


# ============ LSTM TRAINING & PREDICTION ============
def train_lstm(df):
    """Train LSTM model"""
    prices = df["Close"].values[-500:].reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(config.LOOKBACK, len(scaled)):
        X.append(scaled[i-config.LOOKBACK:i])
        y.append(scaled[i])
    
    if len(X) < 50:
        return None, scaler
    
    X, y = np.array(X), np.array(y)
    model = build_lstm_model((config.LOOKBACK, 1), config.HIDDEN_SIZE)
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X, y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=0,
        validation_split=config.VALIDATION_SPLIT
    )
    return model, scaler


def lstm_predict(model, scaler, df):
    """Predict next price using LSTM"""
    if model is None:
        return df["Close"].iloc[-1]
    seq = scaler.transform(df["Close"].values[-config.LOOKBACK:].reshape(-1, 1))
    pred_scaled = model.predict(seq.reshape(1, config.LOOKBACK, 1), verbose=0)[0, 0]
    return scaler.inverse_transform([[pred_scaled]])[0, 0]


# ============ RANDOM FOREST CLASSIFIER ============
def create_labels(df, horizon=None):
    """Create trading labels (BUY/HOLD/SELL)"""
    if horizon is None:
        horizon = config.TRADING_HORIZON
    fut = df["Close"].shift(-horizon)
    ret = (fut - df["Close"]) / df["Close"]
    labels = pd.Series("HOLD", index=df.index)
    labels[ret > 0.02] = "BUY"
    labels[ret < -0.02] = "SELL"
    return labels


def build_features(df):
    """Build features for ML model"""
    feats = pd.DataFrame(index=df.index)
    feats["sma_signal"] = (df["SMA50"] > df["SMA200"]).astype(int)
    feats["ema_signal"] = (df["EMA50"] > df["EMA200"]).astype(int)
    feats["rsi"] = df["RSI"].fillna(50)
    feats["macd"] = df["MACD"].fillna(0)
    feats["bb_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 0.001)
    feats["roc_5"] = df["roc_5"].fillna(0)
    feats["roc_10"] = df["roc_10"].fillna(0)
    return feats.fillna(0)


def train_rf(features, labels):
    """Train Random Forest classifier"""
    data = features.join(labels.rename("label")).dropna()
    if len(data) < 100:
        return None, None, None, None, None
    
    X, y = data.drop("label", axis=1), data["label"]
    
    # Balance classes
    min_samples = min(y.value_counts().values)
    balanced = pd.concat([
        data[data.label == c].sample(n=min(min_samples, 200), random_state=42)
        for c in y.unique()
    ])
    
    Xb, yb = balanced.drop("label", axis=1), balanced["label"]
    Xtr, Xte, ytr, yte = train_test_split(
        Xb, yb,
        stratify=yb,
        test_size=config.RF_TEST_SIZE,
        random_state=42
    )
    
    clf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)
    return clf, Xtr, Xte, ytr, yte


# ============ RECOMMENDATION SYSTEM ============
def get_stock_recommendation(ticker, df, model, scaler, clf, features):
    """Generate stock recommendation based on all models"""
    try:
        recommendation = clf.predict(features.iloc[[-1]])[0]
        confidence = clf.predict_proba(features.iloc[[-1]]).max() * 100
        
        current_price = df['Close'].iloc[-1]
        lstm_price = lstm_predict(model, scaler, df)
        price_change = ((lstm_price - current_price) / current_price) * 100
        details = f"Current: ₹{current_price:.2f} | Predicted: ₹{lstm_price:.2f} | Change: {price_change:+.1f}%"
        
        return recommendation, confidence, details
    except Exception as e:
        return "HOLD", 50.0, f"Error: {str(e)[:50]}"


# ============ BACKTESTING ============
def portfolio_backtest(df, signals, start_capital=1.0):
    """Run portfolio backtest"""
    cash = start_capital
    position = 0
    portfolio_values = []
    trade_returns = []
    entry_price = None
    trades = []

    for date, signal in signals.items():
        if date not in df.index:
            continue
        price = df.loc[date, 'Close']
        
        if signal == 'BUY' and position == 0:
            position = cash / price
            cash = 0
            entry_price = price
            trades.append({'date': date, 'action': 'BUY', 'price': price})
        elif signal == 'SELL' and position > 0:
            cash = position * price
            position = 0
            if entry_price:
                trade_returns.append((price - entry_price) / entry_price)
            trades.append({'date': date, 'action': 'SELL', 'price': price})
            entry_price = None
        
        portfolio_values.append((date, cash + position * price))

    if position > 0 and entry_price is not None:
        final_price = df['Close'].iloc[-1]
        trade_returns.append((final_price - entry_price) / entry_price)

    portfolio_df = pd.DataFrame(portfolio_values, columns=['Date', 'PortfolioValue']).set_index('Date')
    
    if len(portfolio_df) > 0:
        returns = portfolio_df['PortfolioValue'].pct_change().dropna()
        cumulative_return = portfolio_df['PortfolioValue'].iloc[-1] / portfolio_df['PortfolioValue'].iloc[0] - 1
        years = max((portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25, 0.1)
        annualized_return = (1 + cumulative_return) ** (1 / years) - 1
        sharpe_ratio = (annualized_return / (returns.std() * np.sqrt(252))) if returns.std() != 0 else 0
        
        summary = {
            'start_date': portfolio_df.index[0].date(),
            'end_date': portfolio_df.index[-1].date(),
            'cumulative_return': cumulative_return * 100,
            'annualized_return': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades)
        }
    else:
        summary = None
    
    return portfolio_df, trades, trade_returns, summary
