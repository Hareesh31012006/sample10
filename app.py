import streamlit as st
import yfinance as yf
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Title
# ---------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Prediction + Sentiment (LSTM with EPOCHS)") 
st.caption("Yahoo Finance + Google News + HuggingFace + PyTorch — No API Keys Needed ✅")


# ---------------------------
# Sidebar Inputs
# ---------------------------
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y"], 1)
epochs = st.sidebar.slider("Epochs (Training)", 5, 300, 80, 5)
seq_len = st.sidebar.slider("Sequence Length (Days)", 10, 120, 30, 5)

run = st.sidebar.button("Start Training")


# ---------------------------
# Sentiment Model
# ---------------------------
hf = pipeline("sentiment-analysis")


# ---------------------------
# Helper Functions
# ---------------------------
def load_stock(symbol):
    return yf.download(symbol, period=period)[["Open","High","Low","Close","Volume"]].dropna()

def load_news(symbol):
    g = GNews(language="en", max_results=20)
    data = g.get_news(symbol)
    return [n["title"] + " " + (n.get("description") or "") for n in data]

def get_sentiment(text):
    tb = TextBlob(text).sentiment.polarity
    label = hf(text[:512])[0]["label"]
    hf_score = 1 if label=="POSITIVE" else -1 if label=="NEGATIVE" else 0
    return (tb + hf_score) / 2

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 3]) # close price index
    return np.array(X), np.array(y).reshape(-1,1)

class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])


# ---------------------------
# Main Logic
# ---------------------------
if run:
    st.subheader(f"📊 Stock Data: {symbol}")

    df = load_stock(symbol)
    st.line_chart(df["Close"])

    # ------------------ Sentiment ------------------
    st.subheader("📰 News Sentiment")
    news = load_news(symbol)

    if news:
        scores = [get_sentiment(t) for t in news]
        avg_sent = np.mean(scores)
        st.write(f"**Avg Sentiment Score:** {avg_sent:.3f}")

        fig, ax = plt.subplots()
        sns.histplot(scores, bins=20, kde=True)
        st.pyplot(fig)
    else:
        avg_sent = 0
        st.warning("No news found")

    # ------------------ Data Prep ------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequences(scaled, seq_len)
    split = int(len(X)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # ------------------ Model ------------------
    model = LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    st.subheader(f"🤖 Training LSTM for {epochs} Epochs")
    progress = st.progress(0)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        progress.progress((epoch+1)/epochs)

        st.write(f"Epoch {epoch+1}/{epochs} - Loss = {loss.item():.6f}")

    # Plot loss curve
    fig2, ax2 = plt.subplots()
    ax2.plot(loss_history)
    ax2.set_title("Training Loss Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("MSE Loss")
    st.pyplot(fig2)

    # ------------------ Evaluation ------------------
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    st.subheader("📌 Model Test Performance")
    st.write(f"RMSE: `{rmse:.4f}`")
    st.write(f"MAE: `{mae:.4f}`")
    st.write(f"R² Score: `{r2:.4f}`")

    # ------------------ Next Day Prediction ------------------
    last_seq = torch.tensor(scaled[-seq_len:]).float().unsqueeze(0)
    with torch.no_grad():
        predicted_scaled = model(last_seq).item()

    dummy = np.zeros((1,5))
    dummy[0][3] = predicted_scaled
    next_price = scaler.inverse_transform(dummy)[0][3]

    last_price = df["Close"].iloc[-1]
    pct = (next_price-last_price)/last_price*100

    st.subheader("📈 Next Day Forecast")
    st.metric("Last Close", f"${last_price:.2f}")
    st.metric("Predicted Close", f"${next_price:.2f}", f"{pct:.2f}%")

    signal = "BUY ✅" if next_price > last_price and avg_sent > 0 else "SELL ❌" if next_price < last_price else "HOLD ⚠️"
    st.success(f"**Signal: {signal}**")
