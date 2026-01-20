import streamlit as st
st.write("Theme loaded:", st.config.get_option("theme.primaryColor"))

import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import os
import time
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import sqlite3
from datetime import datetime as dt, timedelta, date
import io
import feedparser
import requests

# --------------------
# 🔔 TELEGRAM ALERT CONFIG
# --------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def already_alerted_today(ticker):
    conn = sqlite3.connect("alerts.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            ticker TEXT,
            alert_date TEXT
        )
    """)
    today = date.today().isoformat()
    c.execute(
        "SELECT 1 FROM alerts WHERE ticker=? AND alert_date=?",
        (ticker, today)
    )
    exists = c.fetchone() is not None
    conn.close()
    return exists

def mark_alerted_today(ticker):
    conn = sqlite3.connect("alerts.db")
    c = conn.cursor()
    today = date.today().isoformat()
    c.execute(
        "INSERT INTO alerts VALUES (?, ?)",
        (ticker, today)
    )
    conn.commit()
    conn.close()

# --------------------
# App Setup
# --------------------
st.set_page_config(page_title="Growlio 📈", layout="wide")

# --------------------
# Shared: API Keys + clients
# --------------------
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
st.sidebar.write("🔑 OpenAI key loaded:", "✅ Yes" if openai_key else "❌ No")

client = OpenAI(api_key=openai_key) if openai_key else None

has_gcp = "gcp_service_account" in st.secrets
st.sidebar.write("🔒 Google Sheets loaded:", "✅" if has_gcp else "❌")

# --------------------
# Shared helpers
# --------------------
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
    return data

def fetch_news_auto(ticker):
    query = f"{ticker}+stock"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    return [{
        "title": e.title,
        "url": e.link,
        "date": e.published if "published" in e else "Unknown"
    } for e in feed.entries[:8]]

def update_google_sheet_with_news(sheet_key, tickers):
    if not has_gcp:
        return
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(sheet_key).sheet1
    rows = []
    for t in tickers:
        for n in fetch_news_auto(t):
            rows.append([t, n["title"], n["url"], n["date"]])
    if rows:
        ws.clear()
        ws.update([["ticker", "title", "url", "date"]] + rows)

def openai_summary_from_headlines(ticker, headlines):
    if not client:
        return "OpenAI key missing."
    combined = " | ".join(headlines[:12])
    prompt = f"Explain why {ticker} moved today based on: {combined}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

# --------------------
# Growlio Page (UNCHANGED UI + ALERTS ADDED)
# --------------------
def growlio_page():
    st.title("📊 Growlio - Investment Learning App")

    st.sidebar.header("Stock Settings (Growlio)")
    tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma separated)", "AAPL, MSFT, TSLA")
    start = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.date.today())
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    data = load_data(tickers, start, end)

    st.subheader("📈 Stock Metrics")
    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        last_close = data[ticker]["Close"].iloc[-1]
        first_close = data[ticker]["Close"].iloc[0]
        change = ((last_close - first_close) / first_close) * 100
        cols[i].metric(ticker, f"${last_close:.2f}", f"{change:.2f}%")

    st.subheader("📉 Stock Price Comparison")
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=data[ticker].index, y=data[ticker]["Close"], mode="lines", name=ticker))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Google Sheet Settings")
    sheet_key = st.text_input(
        "Google Sheet ID (Sheet key)",
        value="10zj6tfkdwxNH9lPDeAx5QdM-vcx3G_FsICpC6Us8dx8"
    )

    if st.button("🔄 Refresh News for All Tickers"):
        update_google_sheet_with_news(sheet_key, tickers)

    st.subheader("🔍 Detailed Analysis per Stock")
    for ticker in tickers:
        st.markdown(f"## {ticker}")
        df = data[ticker].copy()

        df["50MA"] = df["Close"].rolling(50).mean()
        df["200MA"] = df["Close"].rolling(200).mean()
        df["Signal"] = (
            (df["50MA"] > df["200MA"]) &
            (df["50MA"].shift(1) <= df["200MA"].shift(1))
        )

        buy_signals = df[df["Signal"]]

        # 🔔 ALERT LOGIC (NON-INTRUSIVE)
        if not buy_signals.empty:
            last_signal_date = buy_signals.index[-1].date()
            if last_signal_date == date.today():
                if not already_alerted_today(ticker):
                    price = df.loc[buy_signals.index[-1], "Close"]
                    send_telegram_alert(
                        f"🚀 BUY SIGNAL\nTicker: {ticker}\nPrice: ${price:.2f}\nStrategy: Golden Cross"
                    )
                    mark_alerted_today(ticker)

        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"]
        ))
        fig2.add_trace(go.Scatter(x=df.index, y=df["50MA"], name="50MA"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["200MA"], name="200MA"))
        fig2.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals["Close"],
            mode="markers", marker=dict(symbol="triangle-up", color="green", size=10)
        ))
        st.plotly_chart(fig2, use_container_width=True)

# --------------------
# Navigation (UNCHANGED)
# --------------------
st.sidebar.title("Growlio Super-App")
page = st.sidebar.radio(
    "Select page",
    ["Growlio (default)", "Portfolio Risk Dashboard", "TradeFlow Analyzer"]
)

if page == "Growlio (default)":
    growlio_page()
