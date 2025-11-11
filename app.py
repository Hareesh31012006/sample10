"""
All-in-One Stock Research Suite
- Streamlit dashboard (front-end)
- Flask /predict API (background thread)
- Data APIs: Yahoo Finance + Alpha Vantage
- News: NewsAPI + X (Twitter via Tweepy) + GNews fallback (no key)
- Sentiment: TextBlob + HuggingFace (tries FinBERT first)
- ML: PyTorch LSTM or TensorFlow LSTM (user selectable)
- Train/Test split with EPOCHS, LR, BATCH SIZE
- Metrics: RMSE, MAE, R² + loss curve
- Backtesting: Backtrader (optional)
- Database (optional toggles): PostgreSQL for prices, MongoDB for news

ENV VARS (optional but supported):
  ALPHA_VANTAGE_API_KEY, NEWSAPI_KEY, X_BEARER_TOKEN,
  POSTGRES_DSN, MONGO_URI

RUN:
  pip install -r requirements.txt
  python -m textblob.download_corpora
  streamlit run app.py
"""

# ---- HuggingFace / TensorFlow fix for Python 3.13 ----
from transformers import pipeline
# ------------------- Python 3.13 HuggingFace Fix -------------------
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# -------------------------------------------------------------------

# ------------------------------------------------------

import threading
from datetime import datetime
from typing import List, Dict, Tuple

# ===== UI & Viz =====
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Data =====
import numpy as np
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

# ===== News & NLP =====
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline

# ===== ML: PyTorch & TensorFlow =====
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ===== Optional: APIs & DBs =====
try:
    from newsapi import NewsApiClient
    NEWS_AVAILABLE = True
except Exception:
    NEWS_AVAILABLE = False

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except Exception:
    TWEEPY_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    PG_AVAILABLE = True
except Exception:
    PG_AVAILABLE = False

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except Exception:
    MONGO_AVAILABLE = False

try:
    import backtrader as bt
    BT_AVAILABLE = True
except Exception:
    BT_AVAILABLE = False

# ===== Config & Secrets =====
def cfg(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

AV_KEY       = cfg("ALPHA_VANTAGE_API_KEY")
NEWS_KEY     = cfg("NEWSAPI_KEY")
X_BEARER     = cfg("X_BEARER_TOKEN")
POSTGRES_DSN = cfg("POSTGRES_DSN")
MONGO_URI    = cfg("MONGO_URI")

# ===== App Header =====
st.set_page_config(page_title="All Tools Stock Suite", layout="wide")
st.title("📈 Full-Stack Stock Suite — All Tools + Epoch Training/Testing")
st.caption("Yahoo · AlphaVantage · NewsAPI · X · TextBlob · HF · PyTorch/TensorFlow · Backtrader · Postgres · MongoDB")

# ===== HF pipeline (FinBERT -> fallback) =====
@st.cache_resource
def get_hf_pipe():
    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=device)
    except Exception:
        return pipeline("sentiment-analysis", device=device)

HF = get_hf_pipe()

# ===== External clients =====
@st.cache_resource
def get_clients():
    news_cli = None
    x_cli = None
    if NEWS_AVAILABLE and NEWS_KEY:
        try:
            news_cli = NewsApiClient(api_key=NEWS_KEY)
        except Exception:
            news_cli = None
    if TWEEPY_AVAILABLE and X_BEARER:
        try:
            x_cli = tweepy.Client(bearer_token=X_BEARER, wait_on_rate_limit=True)
        except Exception:
            x_cli = None
    pg = None
    mongo = None
    if PG_AVAILABLE and POSTGRES_DSN:
        try:
            pg = psycopg2.connect(POSTGRES_DSN)
        except Exception:
            pg = None
    if MONGO_AVAILABLE and MONGO_URI:
        try:
            mongo = MongoClient(MONGO_URI)["stock_news"]
        except Exception:
            mongo = None
    return news_cli, x_cli, pg, mongo

NEWS_CLI, X_CLI, PG_CONN, MONGO_DB = get_clients()

# ===== Flask API in background =====
from flask import Flask, request, jsonify
api = Flask(__name__)
SHARED = {"pred": None, "sent": None, "sym": None}

@api.get("/predict")
def predict_endpoint():
    sym = request.args.get("symbol", "AAPL").upper()
    try:
        df = get_market_data(sym, source="yahoo", period="1y", interval="1d")
        texts = fetch_news(sym, use_newsapi=True, use_x=True)
        s = np.mean([combined_sent(t) for t in texts]) if texts else 0.0
        pred, _ = train_and_predict(df, framework="pytorch", epochs=10, seq_len=30, batch_size=64, lr=1e-3)
        SHARED.update({"pred": float(pred), "sent": float(s), "sym": sym})
        return jsonify(SHARED)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_api():
    api.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_api, daemon=True).start()

# ===== Data fetchers =====

def get_market_data(symbol: str, source: str = "yahoo", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    if source == "yahoo":
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("No data from Yahoo.")
        df = df.rename(columns=str.title).dropna()
        return df[["Open","High","Low","Close","Volume"]]
    # Alpha Vantage daily fallback
    if source == "alphavantage":
        if not AV_KEY:
            raise RuntimeError("ALPHA_VANTAGE_API_KEY missing.")
        ts = TimeSeries(key=AV_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
        data = data.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"})
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        for c in ["Open","High","Low","Close","Volume"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        return data.dropna()
    raise ValueError("Unknown source")


def fetch_news(query: str, use_newsapi: bool = True, use_x: bool = True, gnews_fallback: bool = True) -> List[str]:
    texts: List[str] = []
    if use_newsapi and NEWS_CLI:
        try:
            r = NEWS_CLI.get_everything(q=query, language="en", sort_by="publishedAt", page_size=50)
            for a in r.get("articles", []):
                title = a.get("title", "")
                desc = a.get("description", "")
                t = (title + " " + str(desc)).strip()
                if t:
                    texts.append(t)
        except Exception:
            pass
    if use_x and X_CLI:
        try:
            res = X_CLI.search_recent_tweets(query=query, max_results=50)
            if res and res.data:
                texts.extend([t.text for t in res.data if t and t.text])
        except Exception:
            pass
    if gnews_fallback and not texts:
        try:
            g = GNews(language="en", max_results=20)
            news = g.get_news(query)
            texts.extend([(n.get("title","") + " " + (n.get("description") or "")).strip() for n in news])
        except Exception:
            pass
    return [t for t in texts if t]

# ===== Sentiment =====

def _map_finbert(payload) -> float:
    if not isinstance(payload, list) or not payload:
        return 0.0
    best = max(payload, key=lambda d: d.get("score", 0))
    lab = (best.get("label", "") or "").lower()
    if "pos" in lab: return 1.0
    if "neg" in lab: return -1.0
    return 0.0


def combined_sent(text: str) -> float:
    try:
        tb = TextBlob(text).sentiment.polarity
    except Exception:
        tb = 0.0
    try:
        res = HF(text[:512])
        if isinstance(res, list) and res and isinstance(res[0], dict) and "label" in res[0]:
            # generic SA
            lab = res[0]["label"].upper()
            hf = 1.0 if lab.startswith("POS") else -1.0 if lab.startswith("NEG") else 0.0
        else:
            hf = _map_finbert(res)
    except Exception:
        hf = 0.0
    return float((tb + hf) / 2.0)

# ===== Supervised windowing =====

def make_supervised(df: pd.DataFrame, seq_len: int = 30, features: List[str] = ["Open","High","Low","Close","Volume" ], target: str = "Close"):
    df = df.copy().dropna()
    scaler = MinMaxScaler()
    A = scaler.fit_transform(df[features].astype(float))
    tgt_idx = features.index(target)
    X, y = [], []
    for i in range(len(A) - seq_len):
        X.append(A[i:i+seq_len])
        y.append(A[i+seq_len, tgt_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1,1)
    return X, y, scaler, tgt_idx

# ===== Models =====
class TorchLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, layers_n: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers_n, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:])


def train_pytorch(X_train, y_train, X_test, y_test, epochs=40, lr=1e-3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchLSTM(X_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train).to(device))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    loss_hist = []
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        loss_hist.append(loss.item())
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test).to(device)).cpu().numpy()
    metrics = {
        "rmse": mean_squared_error(y_test, preds, squared=False),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }
    return model, preds, metrics, loss_hist


def train_tensorflow(X_train, y_train, X_test, y_test, epochs=40, lr=1e-3, batch_size=64):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.1),
        layers.LSTM(64),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_test, verbose=0)
    metrics = {
        "rmse": mean_squared_error(y_test, preds, squared=False),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }
    loss_hist = hist.history.get("loss", [])
    return model, preds, metrics, loss_hist

# ===== Unified training helper =====

def train_and_predict(df: pd.DataFrame, framework: str, epochs: int, seq_len: int, batch_size: int, lr: float):
    feats = ["Open","High","Low","Close","Volume"]
    X, y, scaler, close_idx = make_supervised(df, seq_len=seq_len, features=feats, target="Close")
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if framework == "pytorch":
        model, preds, metrics, loss_hist = train_pytorch(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, batch_size=batch_size)
        last_seq = X[-1:]
        with torch.no_grad():
            nxt = model(torch.tensor(last_seq)).cpu().numpy()[0,0]
    else:
        model, preds, metrics, loss_hist = train_tensorflow(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, batch_size=batch_size)
        last_seq = X[-1:]
        nxt = float(model.predict(last_seq, verbose=0)[0,0])

    dummy = np.zeros((1, len(feats)))
    dummy[0, close_idx] = nxt
    next_price = MinMaxScaler().fit(df[feats].astype(float)).inverse_transform(dummy)[0, close_idx]  # placeholder inverse; recalc with scaler
    # Correct inverse using earlier scaler
    next_price = scaler.inverse_transform(dummy)[0, close_idx]

    return next_price, metrics, loss_hist

# ===== DB persistence =====

def save_prices_postgres(df: pd.DataFrame, symbol: str):
    if not (PG_CONN and PG_AVAILABLE):
        return
    try:
        with PG_CONN.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT,
                    dt TIMESTAMP,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION
                );
            """)
            rows = [(symbol, idx.to_pydatetime(), float(r.Open), float(r.High), float(r.Low), float(r.Close), float(r.Volume)) for idx, r in df.iterrows()]
            execute_batch(cur, "INSERT INTO prices (symbol, dt, open, high, low, close, volume) VALUES (%s,%s,%s,%s,%s,%s,%s)", rows, page_size=500)
        PG_CONN.commit()
    except Exception:
        PG_CONN.rollback()


def save_news_mongo(texts: List[str], symbol: str):
    if not (MONGO_DB and MONGO_AVAILABLE):
        return
    try:
        MONGO_DB.news.insert_one({"symbol": symbol, "ts": datetime.utcnow(), "articles": texts})
    except Exception:
        pass

# ===== Backtest (simple signal) =====
if BT_AVAILABLE:
    class SignalStrategy(bt.Strategy):
        params = dict(thresh=0.0)
        def __init__(self):
            self.signal = self.datas[0].lines.signal
        def next(self):
            sig = self.signal[0]
            if not self.position:
                if sig > self.p.thresh: self.buy()
                elif sig < -self.p.thresh: self.sell()
            else:
                if (self.position.size>0 and sig< -self.p.thresh) or (self.position.size<0 and sig> self.p.thresh):
                    self.close()


def run_backtest(df: pd.DataFrame, sig: pd.Series, cash: float = 10000.0):
    if not BT_AVAILABLE:
        return None
    data = df.copy()[["Open","High","Low","Close","Volume"]]
    data = data.assign(signal=sig.reindex(data.index).fillna(0.0))
    class PandasFeed(bt.feeds.PandasData):
        lines = ("signal",)
        params = (("datetime", None),("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume"),("openinterest",None),("signal","signal"))
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    cerebro.addstrategy(SignalStrategy, thresh=0.0)
    cerebro.adddata(PandasFeed(dataname=data))
    cerebro.run()
    return {"start_cash": cash, "final_value": cerebro.broker.getvalue()}

# ===== Sidebar controls =====
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol", "AAPL").upper()
    data_source = st.selectbox("Price Source", ["yahoo", "alphavantage"], help="Alpha Vantage needs API key")
    period = st.selectbox("Period", ["6mo","1y","2y","5y"], index=2)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)

    st.markdown("---")
    st.subheader("ML Training")
    framework = st.selectbox("Framework", ["pytorch", "tensorflow" if TF_AVAILABLE else "tensorflow (unavailable)"], index=0)
    epochs = st.slider("Epochs", 5, 300, 80, 5)
    seq_len = st.slider("Sequence length", 10, 120, 30, 5)
    batch_size = st.slider("Batch size", 16, 256, 64, 16)
    lr = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)

    st.markdown("---")
    st.subheader("News Sources")
    use_newsapi = st.checkbox("Use NewsAPI", value=bool(NEWS_CLI))
    use_x = st.checkbox("Use X/Twitter", value=bool(X_CLI))
    gnews_fb = st.checkbox("Allow GNews fallback", value=True)

    st.markdown("---")
    persist_pg = st.checkbox("Save prices to PostgreSQL", value=False)
    persist_mongo = st.checkbox("Save news to MongoDB", value=False)

    run_btn = st.button("Run Analysis")

# ===== Main run =====
if run_btn:
    try:
        with st.spinner("Fetching prices..."):
            df = get_market_data(symbol, source=data_source, period=period, interval=interval)
        st.success(f"Loaded {len(df):,} rows for {symbol}")
        st.line_chart(df[["Close"]])

        if persist_pg:
            save_prices_postgres(df, symbol)

        with st.spinner("Fetching news/tweets..."):
            texts = fetch_news(symbol, use_newsapi=use_newsapi, use_x=use_x, gnews_fallback=gnews_fb)
        if texts:
            if persist_mongo:
                save_news_mongo(texts, symbol)
            scores = [combined_sent(t) for t in texts]
            avg_sent = float(np.mean(scores)) if scores else 0.0
            col1, col2 = st.columns(2)
            col1.metric("Avg Sentiment", f"{avg_sent:.3f}")
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(scores, bins=20, kde=True, ax=ax)
            ax.set_title("News/Tweet sentiment")
            st.pyplot(fig)
        else:
            avg_sent = 0.0
            st.info("No news/tweets fetched (check keys or enable GNews fallback)")

        # Train/Test with epochs
        st.subheader("🤖 Training & Testing")
        with st.spinner(f"Training {framework} model for {epochs} epochs..."):
            fw = "pytorch" if framework.startswith("pytorch") else "tensorflow"
            next_price, metrics, loss_hist = train_and_predict(df, framework=fw, epochs=epochs, seq_len=seq_len, batch_size=batch_size, lr=lr)
        st.write({k.upper(): round(v,6) for k,v in metrics.items()})

        # Loss curve
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(loss_hist)
        ax2.set_title("Training Loss per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        st.pyplot(fig2)

        last_close = float(df["Close"].iloc[-1])
        pct = (next_price - last_close) / (last_close + 1e-9) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Next Close", f"{next_price:,.2f}", f"{pct:.2f}% vs last")
        c2.metric("Last Close", f"{last_close:,.2f}")
        c3.metric("Avg Sentiment", f"{avg_sent:.3f}")

        signal = "HOLD"
        if next_price > last_close and avg_sent > 0: signal = "BUY"
        elif next_price < last_close and avg_sent < 0: signal = "SELL"
        st.info(f"Signal: **{signal}**")

        # Quick backtest (demo signal from 5-day momentum + sentiment)
        try:
            if BT_AVAILABLE:
                mom = df["Close"].pct_change().rolling(5).mean().fillna(0)
                sig_series = np.sign(mom + (avg_sent * 0.01))
                bt_res = run_backtest(df, sig=pd.Series(sig_series, index=df.index), cash=10000)
                if bt_res:
                    st.subheader("Backtest (demo)")
                    st.write(bt_res)
        except Exception as e:
            st.caption(f"Backtest skipped: {e}")

        SHARED.update({"pred": float(next_price), "sent": float(avg_sent), "sym": symbol})

    except Exception as e:
        st.error(f"Run failed: {e}")

st.markdown("---")
st.caption("Tip: Set API keys in .streamlit/secrets.toml or environment. App gracefully falls back when keys are missing.")
