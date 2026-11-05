import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NSE Stock Analysis Terminal Pro",
    page_icon="📈",
    layout="wide"
)

st_autorefresh(interval=10000, key="nse_refresh")

RUPEE = "₹"

st.markdown("""
<style>
.stApp {background:#ffffff;color:#111827}
.block-container {max-width:1600px;padding-top:1.2rem}
[data-testid="stSidebar"] {background:#f8fafc}
[data-testid="stMetric"] {
    background:white;border:1px solid #e5e7eb;
    padding:13px;border-radius:12px;
}
[data-testid="stMetricValue"] {color:#111827}
h1,h2,h3,p,label {color:#111827}
.stTabs [data-baseweb="tab-list"] {
    gap:5px;background:#f8fafc;padding:6px;border-radius:10px;
}
.stTabs [aria-selected="true"] {
    background:#dcfce7!important;color:#15803d!important;
}
.stButton>button,.stDownloadButton>button {
    background:#16a34a;color:white;border-radius:8px;
    border:1px solid #16a34a;
}
.stButton>button:hover,.stDownloadButton>button:hover {
    background:#15803d;color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE STOCK ANALYSIS TERMINAL PRO")
st.caption(
    "Advanced stock research | Technical analysis | Portfolio | "
    "Risk | Alerts | Backtesting | Market analytics"
)

st.info(
    "Educational research tool only. Market data may be delayed or incomplete. "
    "Simulations and indicators do not guarantee future performance."
)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "alerts": [],
    "watchlist": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"],
    "alert_history": [],
    "portfolio": [],
    "last_alert_prices": {}
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ticker(symbol):
    symbol = symbol.strip().upper()
    if symbol.endswith((".NS", ".BO")) or symbol.startswith("^"):
        return symbol
    return symbol + ".NS"


def money(value):
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return "N/A"
        return f"{RUPEE}{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


@st.cache_data(ttl=600, show_spinner=False)
def history(symbol, period="2y"):
    try:
        data = yf.Ticker(ticker(symbol)).history(
            period=period,
            auto_adjust=True
        )
        if data is None:
            return pd.DataFrame()
        return data.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=8, show_spinner=False)
def live_price(symbol):
    try:
        data = yf.Ticker(ticker(symbol)).history(
            period="1d",
            interval="1m",
            auto_adjust=True
        )
        if not data.empty:
            data = data.dropna(subset=["Close"])
            if not data.empty:
                return float(data.Close.iloc[-1])
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals(symbol):
    try:
        return yf.Ticker(ticker(symbol)).get_info()
    except Exception:
        try:
            return yf.Ticker(ticker(symbol)).info
        except Exception:
            return {}


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1/period, min_periods=period, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/period, min_periods=period, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)

    result = result.mask((loss == 0) & (gain > 0), 100)
    result = result.mask((gain == 0) & (loss > 0), 0)
    return result


def add_indicators(data):
    d = data.copy()
    c, h, l = d.Close, d.High, d.Low

    for n in [9, 20, 50]:
        d[f"EMA {n}"] = c.ewm(span=n, adjust=False).mean()

    for n in [20, 50, 100, 200]:
        d[f"SMA {n}"] = c.rolling(n).mean()

    d["RSI 14"] = rsi(c)

    d["MACD"] = (
        c.ewm(span=12, adjust=False).mean()
        - c.ewm(span=26, adjust=False).mean()
    )
    d["MACD Signal"] = d.MACD.ewm(span=9, adjust=False).mean()
    d["MACD Histogram"] = d.MACD - d["MACD Signal"]

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()

    d["BB Upper"] = mid + 2 * sd
    d["BB Middle"] = mid
    d["BB Lower"] = mid - 2 * sd

    tr = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)

    d["ATR 14"] = tr.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    up = h.diff()
    down = -l.diff()

    plus = up.where((up > down) & (up > 0), 0)
    minus = down.where((down > up) & (down > 0), 0)

    pdi = 100 * plus.ewm(alpha=1/14, adjust=False).mean() / d["ATR 14"]
    mdi = 100 * minus.ewm(alpha=1/14, adjust=False).mean() / d["ATR 14"]

    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    d["ADX 14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    d["ROC 12"] = c.pct_change(12) * 100

    lowest = l.rolling(14).min()
    highest = h.rolling(14).max()

    d["Stochastic %K"] = (
        100 * (c-lowest) / (highest-lowest).replace(0, np.nan)
    )
    d["Stochastic %D"] = d["Stochastic %K"].rolling(3).mean()

    tp = (h+l+c)/3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x-np.mean(x))), raw=True
    )
    d["CCI 20"] = (tp-sma_tp)/(0.015*mad.replace(0, np.nan))

    d["Williams %R"] = (
        -100*(highest-c)/(highest-lowest).replace(0, np.nan)
    )

    if "Volume" in d:
        direction = np.sign(c.diff()).fillna(0)
        d["OBV"] = (direction*d.Volume.fillna(0)).cumsum()

        money_flow = (
            ((c-l)-(h-c))/(h-l).replace(0, np.nan)
        ) * d.Volume.fillna(0)

        positive = money_flow.clip(lower=0).rolling(14).sum()
        negative = (-money_flow.clip(upper=0)).rolling(14).sum()

        d["MFI 14"] = 100-100/(
            1+positive/negative.replace(0, np.nan)
        )

        d["Volume SMA 20"] = d.Volume.rolling(20).mean()

        d["VWAP 20"] = (
            (tp*d.Volume).rolling(20).sum()
            / d.Volume.rolling(20).sum().replace(0, np.nan)
        )

    else:
        d["OBV"] = np.nan
        d["MFI 14"] = np.nan
        d["Volume SMA 20"] = np.nan
        d["VWAP 20"] = np.nan

    d["Donchian Upper"] = h.rolling(20).max()
    d["Donchian Lower"] = l.rolling(20).min()
    d["Donchian Middle"] = (
        d["Donchian Upper"]+d["Donchian Lower"]
    )/2

    d["Keltner Middle"] = c.ewm(span=20, adjust=False).mean()
    d["Keltner Upper"] = d["Keltner Middle"]+2*d["ATR 14"]
    d["Keltner Lower"] = d["Keltner Middle"]-2*d["ATR 14"]

    d["Ichimoku Tenkan"] = (
        h.rolling(9).max()+l.rolling(9).min()
    )/2

    d["Ichimoku Kijun"] = (
        h.rolling(26).max()+l.rolling(26).min()
    )/2

    d["Ichimoku Span A"] = (
        d["Ichimoku Tenkan"]+d["Ichimoku Kijun"]
    )/2

    d["Ichimoku Span B"] = (
        h.rolling(52).max()+l.rolling(52).min()
    )/2

    hl2 = (h+l)/2
    d["Supertrend Upper"] = hl2+3*d["ATR 14"]
    d["Supertrend Lower"] = hl2-3*d["ATR 14"]

    d["Daily Return %"] = c.pct_change()*100

    return d


def technical_score(row):
    score = 0
    signals = []

    if row["EMA 20"] > row["EMA 50"]:
        score += 1
        signals.append("EMA bullish")
    else:
        score -= 1
        signals.append("EMA bearish")

    if row["MACD"] > row["MACD Signal"]:
        score += 1
        signals.append("MACD bullish")
    else:
        score -= 1
        signals.append("MACD bearish")

    if pd.notna(row["RSI 14"]):
        if row["RSI 14"] > 50:
            score += 1
            signals.append("RSI positive")
        elif row["RSI 14"] < 40:
            score -= 1
            signals.append("RSI weak")

    if pd.notna(row["ADX 14"]) and row["ADX 14"] > 25:
        signals.append("Strong trend")

    if pd.notna(row["SMA 50"]):
        if row["Close"] > row["SMA 50"]:
            score += 1
            signals.append("Above SMA 50")
        else:
            score -= 1
            signals.append("Below SMA 50")

    if score >= 3:
        label = "Bullish"
    elif score <= -2:
        label = "Bearish"
    else:
        label = "Neutral"

    return score, label, signals


def candlestick_patterns(d):
    x = d.copy()

    body = (x.Close-x.Open).abs()
    candle_range = (x.High-x.Low).replace(0, np.nan)
    upper = x.High-x[["Open", "Close"]].max(axis=1)
    lower = x[["Open", "Close"]].min(axis=1)-x.Low

    x["Doji"] = body <= candle_range*0.10

    x["Hammer"] = (
        (lower > body*2)
        & (upper < body*0.5)
        & (body/candle_range < 0.4)
    )

    prev_open = x.Open.shift()
    prev_close = x.Close.shift()

    x["Bullish Engulfing
