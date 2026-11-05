import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="NSE Stock Analysis Terminal", layout="wide")
# Rerun the dashboard every 10 seconds; market-data updates depend on the provider.
st_autorefresh(interval=10_000, key="nse_price_refresh_10s")
st.markdown("""
<style>
:root {
  --bg: #ffffff;
  --panel: #ffffff;
  --border: #e5e7eb;
  --text: #111827;
  --muted: #64748b;
  --green: #16a34a;
  --red: #dc2626;
}
.stApp { background: #ffffff; color: var(--text); }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
[data-testid="stHeader"] { background: rgba(255,255,255,.96); }
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid var(--border); }
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
  background: #ffffff; border: 1px solid var(--border); border-radius: 12px;
  padding: 12px; box-shadow: 0 2px 8px rgba(15,23,42,.05);
}
[data-testid="stMetricValue"] { color: #111827; }
[data-testid="stMetricLabel"], .stCaption, small { color: var(--muted) !important; }
h1, h2, h3, p, label { color: #111827; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 6px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #334155; padding: 8px 12px; }
.stTabs [aria-selected="true"] { background: #e8f5ec !important; color: #15803d !important; }
.stButton > button, .stDownloadButton > button { border: 1px solid #16a34a; border-radius: 8px; color: #ffffff; background: #16a34a; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #15803d; border-color: #15803d; }
[data-baseweb="input"] > div, [data-baseweb="select"] > div,
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { background: #ffffff; border-color: #cbd5e1; color: #111827; }
hr { border-color: #e5e7eb; }
/* Streamlit metric deltas: positive green, negative red */
[data-testid="stMetricDelta"] svg { display: inline-block; }
[data-testid="stMetricDelta"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)
st.title("NSE Stock Analysis Terminal")
st.caption("NSE stock research dashboard | Charts, indicators, portfolio, fundamentals, risk and backtesting")
st.info("Educational tool only. Historical data and simulations cannot guarantee future prices. Not financial advice.")

RUPEE = chr(8377)  # Indian Rupee symbol, generated safely at runtime

def ticker(s):
    s=s.strip().upper()
    return s if s.endswith((".NS",".BO")) or s.startswith("^") else s+".NS"

@st.cache_data(ttl=600, show_spinner=False)
def history(s, period="2y"):
    return yf.Ticker(ticker(s)).history(period=period, auto_adjust=True)

@st.cache_data(ttl=8, show_spinner=False)
def latest_intraday_price(s):
    """Fetch the latest available 1-minute intraday bar from Yahoo Finance."""
    try:
        data = yf.Ticker(ticker(s)).history(period="1d", interval="1m", auto_adjust=True)
        if data is not None and not data.empty:
            data = data.dropna(subset=["Close"])
            if not data.empty:
                return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals(s):
    """Fetch Yahoo Finance profile and annual/quarterly statements with fallbacks."""
    symbol = ticker(s)
    t = yf.Ticker(symbol)
    result = {}

    # Company profile: try both supported access patterns.
    for getter in (lambda: t.get_info(), lambda: t.info):
        try:
            data = getter()
            if isinstance(data, dict):
                result.update({k: v for k, v in data.items() if v is not None})
                if result:
                    break
        except Exception:
            continue

    # Try multiple yfinance statement properties; some versions/providers expose
    # different property names or only quarterly statements.
    statement_sources = {
        "Income statement": ["income_stmt", "financials", "quarterly_income_stmt", "quarterly_financials"],
        "Balance sheet": ["balance_sheet", "quarterly_balance_sheet"],
        "Cash flow": ["cashflow", "cash_flow", "quarterly_cashflow", "quarterly_cash_flow"],
    }
    for label, attrs in statement_sources.items():
        found = False
        for attr in attrs:
            try:
                frame = getattr(t, attr)
                if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
                    continue
                # Latest period is normally the first column in yfinance.
                latest_col = frame.columns[0]
                for metric, value in frame[latest_col].items():
                    if pd.notna(value):
                        try:
                            result[f"{label}:{metric}"] = float(value)
                        except (TypeError, ValueError):
                            result[f"{label}:{metric}"] = str(value)
                found = True
                break
            except Exception:
                continue
    return result

def rsi(series, n=14):
    d=series.diff()
    gain=d.clip(lower=0).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    loss=(-d.clip(upper=0)).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    return 100-100/(1+rs)

def add_indicators(df):
    d=df.copy(); c=d.Close; h=d.High; l=d.Low
    for n in [9,20,50]: d[f"EMA {n}"]=c.ewm(span=n,adjust=False).mean()
    for n in [20,50,200]: d[f"SMA {n}"]=c.rolling(n).mean()
    d["RSI 14"]=rsi(c)
    d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d["MACD Signal"]=d["MACD"].ewm(span=9,adjust=False).mean()
    d["MACD Histogram"]=d["MACD"]-d["MACD Signal"]
    mid=c.rolling(20).mean(); sd=c.rolling(20).std()
    d["BB Upper"]=mid+2*sd; d["BB Middle"]=mid; d["BB Lower"]=mid-2*sd
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0.0); minus=down.where((down>up)&(down>0),0.0)
    pdi=100*plus.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr
    mdi=100*minus.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["ADX 14"]=dx.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    d["ATR 14"]=atr

    # Momentum indicators
    d["ROC 12"] = c.pct_change(12) * 100
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    d["Stochastic %K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    d["Stochastic %D"] = d["Stochastic %K"].rolling(3).mean()
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d["CCI 20"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    highest14 = h.rolling(14).max()
    lowest14 = l.rolling(14).min()
    d["Williams %R"] = -100 * (highest14 - c) / (highest14 - lowest14).replace(0, np.nan)

    # Volume indicators
    direction = np.sign(c.diff()).fillna(0)
    d["OBV"] = (direction * d["Volume"].fillna(0)).cumsum() if "Volume" in d else np.nan
    if "Volume" in d:
        money_flow_mult = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        money_flow_vol = money_flow_mult.fillna(0) * d["Volume"].fillna(0)
        positive_flow = money_flow_vol.clip(lower=0).rolling(14).sum()
        negative_flow = (-money_flow_vol.clip(upper=0)).rolling(14).sum()
        d["MFI 14"] = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.nan)))
        d["VWAP 20"] = (tp * d["Volume"].fillna(0)).rolling(20).sum() / d["Volume"].fillna(0).rolling(20).sum().replace(0, np.nan)
        d["Volume SMA 20"] = d["Volume"].rolling(20).mean()
    else:
        d["MFI 14"] = np.nan
        d["VWAP 20"] = np.nan
        d["Volume SMA 20"] = np.nan

    # Trend / volatility indicators
    d["Donchian Upper"] = h.rolling(20).max()
    d["Donchian Lower"] = l.rolling(20).min()
    d["Donchian Middle"] = (d["Donchian Upper"] + d["Donchian Lower"]) / 2
    d["Keltner Middle"] = c.ewm(span=20, adjust=False).mean()
    d["Keltner Upper"] = d["Keltner Middle"] + 2 * atr
    d["Keltner Lower"] = d["Keltner Middle"] - 2 * atr

    # Ichimoku lines (unshifted for easy historical display)
    d["Ichimoku Tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    d["Ichimoku Kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    d["Ichimoku Span A"] = (d["Ichimoku Tenkan"] + d["Ichimoku Kijun"]) / 2
    d["Ichimoku Span B"] = (h.rolling(52).max() + l.rolling(52).min()) / 2

    # Supertrend-style direction estimate based on ATR bands
    hl2 = (h + l) / 2
    d["Supertrend Upper"] = hl2 + 3 * atr
    d["Supertrend Lower"] = hl2 - 3 * atr
    d["Daily Return %"] = c.pct_change() * 100
    return d

def money(x):
    try: return f"{RUPEE}{float(x):,.2f}" if pd.notna(x) and np.isfinite(float(x)) else "N/A"
    except Exception: return "N/A"

with st.sidebar:
    st.header("Stock selector")
    symbol=st.text_input("NSE ticker","RELIANCE").strip().upper()
    period=st.selectbox("Historical data",["6mo","1y","2y","5y","10y"],index=2)
    chart_days=st.selectbox("Chart window",[60,120,252,500,1000],index=2,format_func=lambda n:f"Last {n} trading days")
    st.caption("Examples: RELIANCE, TCS, INFY, HDFCBANK, SBIN")

if not symbol: st.stop()
try: raw=history(symbol,period)
except Exception as e:
    st.error(f"Could not load stock data: {e}"); st.stop()
if raw is None or raw.empty or not {"Close","Open","High","Low"}.issubset(raw.columns):
    st.error("No price data found. Check the ticker symbol."); st.stop()
raw=raw.dropna(subset=["Close"])
if len(raw)<30: st.error("Not enough historical data."); st.stop()
df=add_indicators(raw); close=df.Close; price=float(close.iloc[-1]); ret=close.pct_change().dropna()
# Prefer the latest available intraday quote for the displayed/current price.
intraday_price = latest_intraday_price(symbol)
if intraday_price is not None and np.isfinite(intraday_price):
    price = intraday_price
info=fundamentals(symbol)

tabs=st.tabs(["Overview", "Clean Chart", "Indicators", "Growth / Loss", "Risk", "Compare", "Watchlist", "Portfolio", "Fundamentals", "News", "Backtesting"])

with tabs[0]:
    st.subheader(f"{symbol} overview")
    prev=float(close.iloc[-1]); chg=(price/prev-1)*100 if prev else np.nan
    c=st.columns(4)
    c[0].metric("Latest available price",money(price),f"{chg:+.2f}% vs previous close", delta_color="normal")
    c[1].metric("RSI (14)",f"{df['RSI 14'].iloc[-1]:.2f}" if pd.notna(df["RSI 14"].iloc[-1]) else "N/A")
    c[2].metric("MACD histogram",f"{df['MACD Histogram'].iloc[-1]:.3f}")
    c[3].metric("ADX (14)",f"{df['ADX 14'].iloc[-1]:.2f}" if pd.notna(df["ADX 14"].iloc[-1]) else "N/A")
    votes=sum([df["EMA 20"].iloc[-1]>df["EMA 50"].iloc[-1],df["MACD Histogram"].iloc[-1]>0,
               pd.notna(df["RSI 14"].iloc[-1]) and 50<df["RSI 14"].iloc[-1]<70,
               pd.notna(df["SMA 50"].iloc[-1]) and price>df["SMA 50"].iloc[-1]])
    if votes>=3: st.success("Technical indicators currently lean bullish.")
    elif votes<=1: st.error("Technical indicators currently lean bearish.")
    else: st.warning("Technical indicators are mixed / neutral.")
    st.caption("Price refreshes every 10 seconds when updated data is available. Quotes may be delayed by the data provider.")



with tabs[1]:
    st.subheader("Clean chart")
    st.write("Choose which lines you want to display on the chart.")
    overlay_options=["EMA 9","EMA 20","EMA 50","SMA 20","SMA 50","SMA 200",
                     "BB Upper","BB Middle","BB Lower","VWAP 20",
                     "Donchian Upper","Donchian Middle","Donchian Lower",
                     "Keltner Upper","Keltner Middle","Keltner Lower",
                     "Ichimoku Tenkan","Ichimoku Kijun","Ichimoku Span A","Ichimoku Span B",
                     "Supertrend Upper","Supertrend Lower"]
    overlays=st.multiselect("Show indicator lines",overlay_options,default=["EMA 20","SMA 50"])
    chart_type=st.radio("Price style",["Candlestick","Closing price line"],horizontal=True)
    volume=st.checkbox("Show volume",False)
    view=df.tail(chart_days); fig=go.Figure()
    if chart_type=="Candlestick":
        fig.add_trace(go.Candlestick(x=view.index,open=view.Open,high=view.High,low=view.Low,close=view.Close,name="Price", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"))
    else: fig.add_trace(go.Scatter(x=view.index,y=view.Close,mode="lines",name="Close"))
    for col in overlays: fig.add_trace(go.Scatter(x=view.index,y=view[col],mode="lines",name=col))
    if volume and "Volume" in view:
        fig.add_trace(go.Bar(x=view.index,y=view.Volume,name="Volume",yaxis="y2",opacity=.25))
        fig.update_layout(yaxis2=dict(overlaying="y",side="right",showgrid=False))
    fig.update_layout(template="plotly_white",height=570,title=f"{symbol} price chart",xaxis_rangeslider_visible=False,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)


with tabs[2]:
    st.subheader("Separate indicator charts")
    chosen=st.selectbox("Choose indicator",[
        "RSI 14","MACD","ADX 14","ATR 14","Daily return (%)","Bollinger Bands",
        "Stochastic","ROC 12","CCI 20","Williams %R","MFI 14","OBV",
        "VWAP 20","Donchian Channels","Keltner Channels","Ichimoku","Volume"
    ])
    v=df.tail(chart_days); fig=go.Figure()
    if chosen=="MACD":
        fig.add_trace(go.Scatter(x=v.index,y=v["MACD"],name="MACD"))
        fig.add_trace(go.Scatter(x=v.index,y=v["MACD Signal"],name="Signal"))
        fig.add_trace(go.Bar(x=v.index,y=v["MACD Histogram"],name="Histogram"))
    elif chosen=="Bollinger Bands":
        for col in ["BB Upper","BB Middle","BB Lower"]: fig.add_trace(go.Scatter(x=v.index,y=v[col],name=col))
        fig.add_trace(go.Scatter(x=v.index,y=v.Close,name="Close"))
    elif chosen=="Daily return (%)":
        fig.add_trace(go.Bar(x=v.index,y=v["Daily Return %"],name=chosen))
    elif chosen=="Stochastic":
        fig.add_trace(go.Scatter(x=v.index,y=v["Stochastic %K"],name="%K"))
        fig.add_trace(go.Scatter(x=v.index,y=v["Stochastic %D"],name="%D"))
        fig.add_hline(y=80,line_dash="dash"); fig.add_hline(y=20,line_dash="dash")
    elif chosen in ["Donchian Channels","Keltner Channels","Ichimoku"]:
        groups = {
            "Donchian Channels":["Donchian Upper","Donchian Middle","Donchian Lower"],
            "Keltner Channels":["Keltner Upper","Keltner Middle","Keltner Lower"],
            "Ichimoku":["Ichimoku Tenkan","Ichimoku Kijun","Ichimoku Span A","Ichimoku Span B"]
        }
        for col in groups[chosen]:
            fig.add_trace(go.Scatter(x=v.index,y=v[col],name=col))
        fig.add_trace(go.Scatter(x=v.index,y=v.Close,name="Close"))
    elif chosen=="Volume":
        if "Volume" in v:
            fig.add_trace(go.Bar(x=v.index,y=v["Volume"],name="Volume"))
            if "Volume SMA 20" in v:
                fig.add_trace(go.Scatter(x=v.index,y=v["Volume SMA 20"],name="Volume SMA 20"))
    else:
        fig.add_trace(go.Scatter(x=v.index,y=v[chosen],name=chosen))
        if chosen=="RSI 14": st.caption("RSI above 70 is often called overbought; below 30 oversold. Neither is an automatic sell/buy signal."); fig.add_hline(y=70,line_dash="dash"); fig.add_hline(y=30,line_dash="dash")
        if chosen=="ADX 14": fig.add_hline(y=25,line_dash="dash")
        if chosen=="Williams %R": fig.add_hline(y=-20,line_dash="dash"); fig.add_hline(y=-80,line_dash="dash")
        if chosen=="MFI 14": fig.add_hline(y=80,line_dash="dash"); fig.add_hline(y=20,line_dash="dash")
    fig.update_layout(template="plotly_white",height=430,title=chosen,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    cols=["RSI 14","MACD","MACD Signal","MACD Histogram","ADX 14","ATR 14",
          "Stochastic %K","Stochastic %D","ROC 12","CCI 20","Williams %R","MFI 14",
          "OBV","VWAP 20","Volume SMA 20","EMA 9","EMA 20","EMA 50","SMA 20","SMA 50","SMA 200",
          "BB Upper","BB Middle","BB Lower","Donchian Upper","Donchian Middle","Donchian Lower",
          "Keltner Upper","Keltner Middle","Keltner Lower","Ichimoku Tenkan","Ichimoku Kijun",
          "Ichimoku Span A","Ichimoku Span B","Supertrend Upper","Supertrend Lower"]
    st.dataframe(pd.DataFrame({"Indicator":cols,"Latest value":[df[c].iloc[-1] for c in cols]}),use_container_width=True,hide_index=True)


with tabs[3]:
    st.subheader("Future growth and loss scenarios")
    horizon=st.select_slider("Forecast horizon (trading days)",[7,30,90,252],value=30)
    sims=st.select_slider("Simulation paths",[500,1000,2000,5000],value=1000)
    logs=np.log(close/close.shift()).dropna(); mu=float(logs.mean()); sigma=float(logs.std())
    paths=price*np.exp(np.cumsum(np.random.default_rng(42).normal(mu-.5*sigma**2,sigma,(sims,horizon)),axis=1))
    final=paths[:,-1]; up=(final>price).mean()*100; down=(final<price).mean()*100
    p5,p25,p50,p75,p95=np.percentile(final,[5,25,50,75,95])
    c=st.columns(4); c[0].metric("Current price",money(price)); c[1].metric("Simulated outcomes above",f"{up:.1f}%"); c[2].metric("Simulated outcomes below",f"{down:.1f}%"); c[3].metric("Median scenario",money(p50))
    scenario=pd.DataFrame({"Scenario":["Pessimistic 5th percentile","Lower-middle 25th","Median 50th","Upper-middle 75th","Optimistic 95th percentile"],"Price":[p5,p25,p50,p75,p95]})
    scenario["Change (%)"]=(scenario.Price/price-1)*100
    st.dataframe(scenario.style.format({"Price":f"{RUPEE}{{:,.2f}}","Change (%)":"{:+.2f}%"}),use_container_width=True,hide_index=True)
    q5,q50,q95=np.percentile(paths,[5,50,95],axis=0); days=np.arange(1,horizon+1); fig=go.Figure()
    fig.add_trace(go.Scatter(x=days,y=q95,name="95th percentile")); fig.add_trace(go.Scatter(x=days,y=q50,name="Median")); fig.add_trace(go.Scatter(x=days,y=q5,name="5th percentile",fill="tonexty"))
    fig.update_layout(template="plotly_white",height=400,title="Simulated price ranges",xaxis_title="Trading days ahead",yaxis_title=f"Price ({RUPEE})")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Monte Carlo simulation based on historical returns, not a trained AI forecast. Probabilities are not calibrated real-world odds; actual prices may fall outside these scenarios.")


with tabs[4]:
    st.subheader("Risk analysis")
    vol=float(ret.std()*np.sqrt(252)*100); years=len(ret)/252
    ann=((price/float(close.iloc[0]))**(1/years)-1)*100 if years>0 else np.nan
    dd=close/close.cummax()-1; maxdd=float(dd.min()*100); var=float(np.percentile(ret,5)*100)
    sharpe=(ann-6.5)/vol if vol>0 else np.nan
    c=st.columns(4); c[0].metric("Annualized volatility",f"{vol:.2f}%"); c[1].metric("Annualized historical return",f"{ann:.2f}%"); c[2].metric("Max drawdown",f"{maxdd:.2f}%"); c[3].metric("Daily VaR (95%)",f"{var:.2f}%")
    st.metric("Approx. Sharpe ratio (assumed risk-free rate 6.5%)",f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
    dd_df=pd.DataFrame({"Date":dd.index,"Drawdown (%)":dd.values*100}).tail(chart_days)
    st.plotly_chart(px.area(dd_df,x="Date",y="Drawdown (%)",template="plotly_white",title="Drawdown from previous peak"),use_container_width=True)


with tabs[5]:
    st.subheader("Compare stocks")
    syms=st.text_input("2–5 NSE tickers, comma-separated","RELIANCE,TCS,INFY")
    cp=st.selectbox("Comparison period",["6mo","1y","2y","5y"],index=1)
    if st.button("Compare selected stocks"):
        series={}
        for s in list(dict.fromkeys(x.strip().upper() for x in syms.split(",") if x.strip()))[:5]:
            try:
                h=history(s,cp)
                if not h.empty: series[s]=h.Close/h.Close.dropna().iloc[0]*100
            except Exception: pass
        if len(series)>=2:
            frame=pd.DataFrame(series)
            st.plotly_chart(px.line(frame,template="plotly_white",title="Relative performance (start=100)"),use_container_width=True)
            st.dataframe(frame.tail(10),use_container_width=True)
        else: st.warning("Could not retrieve at least two valid stock histories.")


with tabs[6]:
    st.subheader("NSE Watchlist")
    st.caption("Track multiple NSE stocks together. Prices refresh with the dashboard every 10 seconds when the data provider supplies updated quotes.")
    watch_input = st.text_input("Enter NSE tickers separated by commas", "RELIANCE,TCS,INFY,HDFCBANK,SBIN", key="watchlist_symbols")
    if st.button("Update watchlist", key="update_watchlist"):
        st.session_state["watchlist_saved_symbols"] = watch_input
    saved = st.session_state.get("watchlist_saved_symbols", watch_input)
    rows = []
    for item in list(dict.fromkeys(x.strip().upper() for x in saved.split(",") if x.strip()))[:15]:
        try:
            hist = history(item, "5d")
            if hist is None or hist.empty:
                continue
            previous = float(hist["Close"].dropna().iloc[-1])
            live = latest_intraday_price(item)
            current = float(live) if live is not None else previous
            change = ((current / previous) - 1) * 100 if previous else np.nan
            rows.append({"Ticker": item, "Latest available price": money(current), "Change vs previous close (%)": change,
                         "Data": "Intraday quote" if live is not None else "Latest daily close"})
        except Exception:
            continue
    if rows:
        watch_df = pd.DataFrame(rows)
        st.dataframe(watch_df.style.format({"Change vs previous close (%)": "{:+.2f}%"}).map(
            lambda v: "color: #16a34a; font-weight: 600" if isinstance(v, (int, float)) and v > 0 else
                      ("color: #dc2626; font-weight: 600" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["Change vs previous close (%)"]), use_container_width=True, hide_index=True)
    else:
        st.info("No watchlist prices are currently available. Check ticker symbols or try again later.")


with tabs[7]:
    st.subheader("Portfolio profit/loss calculator")
    st.caption("This calculator does not connect to your broker or save transactions.")
    holdings=[]
    for i in range(3):
        with st.expander(f"Holding {i+1}",expanded=(i==0)):
            a,b,c=st.columns(3)
            s=a.text_input("Ticker",["RELIANCE","TCS","INFY"][i],key=f"ps{i}").strip().upper()
            qty=b.number_input("Quantity",min_value=0.0,value=1.0,key=f"pq{i}")
            buy=c.number_input(f"Buy price {RUPEE}",min_value=0.0,value=100.0,key=f"pb{i}")
            if s and qty>0:
                try:
                    h=history(s,"5d")
                    if not h.empty:
                        now=float(h.Close.iloc[-1]); holdings.append({"Symbol":s,"Qty":qty,"Buy price":buy,"Latest price":now,"Invested":qty*buy,"Value":qty*now,"P/L":qty*(now-buy)})
                except Exception: pass
    if holdings:
        pf=pd.DataFrame(holdings); st.dataframe(pf,use_container_width=True,hide_index=True)
        invested=pf.Invested.sum(); value=pf.Value.sum(); c=st.columns(3)
        c[0].metric("Invested",money(invested)); c[1].metric("Latest value",money(value)); c[2].metric("Profit / loss",money(value-invested),f"{(value/invested-1)*100:+.2f}%" if invested else "N/A")


with tabs[8]:
    st.subheader(f"Fundamental analysis: {symbol}")
    st.caption("Displays the latest company profile and financial-statement figures supplied by the data provider. Some metrics may not be reported for every company.")
    profile_fields=[("Company","longName"),("Sector","sector"),("Industry","industry"),("Trailing P/E","trailingPE"),("Forward P/E","forwardPE"),("Price/book","priceToBook"),("EPS","trailingEps"),("ROE","returnOnEquity"),("Profit margin","profitMargins"),("Revenue growth","revenueGrowth"),("Debt/equity","debtToEquity"),("Dividend yield","dividendYield"),("Market cap","marketCap"),("52-week high","fiftyTwoWeekHigh"),("52-week low","fiftyTwoWeekLow")]
    profile_rows=[]
    for label,key in profile_fields:
        val=info.get(key)
        if val is not None:
            if key in ["returnOnEquity","profitMargins","revenueGrowth","dividendYield"]: val=f"{val*100:.2f}%"
            elif key=="marketCap": val=f"{RUPEE}{val:,.0f}"
            profile_rows.append({"Metric":label,"Value":val})
    if profile_rows:
        st.markdown("#### Company profile and valuation")
        st.dataframe(pd.DataFrame(profile_rows),use_container_width=True,hide_index=True)
    statement_groups=[("Income statement", "Income statement:"),("Balance sheet", "Balance sheet:"),("Cash flow", "Cash flow:")]
    statement_found=False
    for title,prefix in statement_groups:
        legacy={"Income statement":"IS:","Balance sheet":"BS:","Cash flow":"CF:"}[title]
        statement_rows=[{"Metric":k[len(prefix):] if k.startswith(prefix) else k[len(legacy):],"Latest reported value":v} for k,v in info.items() if k.startswith(prefix) or k.startswith(legacy)]
        if statement_rows:
            statement_found=True
            st.markdown(f"#### {title}")
            st.dataframe(pd.DataFrame(statement_rows),use_container_width=True,hide_index=True)
    if not profile_rows and not statement_found:
        st.warning("The data provider did not return fundamental data for this symbol. Check the NSE ticker spelling or try again later. This can happen when Yahoo Finance has incomplete coverage for the company.")


with tabs[9]:
    st.subheader(f"News: {symbol}")
    if st.button("Load latest news"):
        try:
            news=yf.Ticker(ticker(symbol)).news or []
            if not news: st.info("No news returned by provider.")
            for item in news[:15]:
                x=item.get("content",item); title=x.get("title","Untitled")
                publisher=(x.get("provider") or {}).get("displayName") or x.get("publisher") or "Unknown publisher"
                url=(x.get("canonicalUrl") or {}).get("url") or x.get("link")
                with st.expander(title):
                    st.write(f"Publisher: {publisher}")
                    if x.get("pubDate"): st.write(f"Published: {x['pubDate']}")
                    if x.get("summary"): st.write(x["summary"])
                    if url: st.link_button("Open article",url)
        except Exception as e: st.error(f"News could not be loaded: {e}")



with tabs[10]:
    st.subheader("🧪 Strategy Backtesting")
    st.write("Test a rule-based strategy on the selected stock's historical data. Signals are generated from completed-day data and positions are applied from the next trading day to reduce look-ahead bias.")
    if len(df) < 80:
        st.warning("Not enough historical data for a meaningful backtest. Select a longer historical period.")
    else:
        strategy = st.selectbox("Strategy to test", [
            "EMA crossover (20/50)", "RSI mean reversion (30/70)",
            "MACD crossover", "Trend confirmation (EMA + MACD + RSI)"
        ])
        direction = st.selectbox("Trading mode", ["Long only", "Long / short (simplified)"])
        fee_bps = st.slider("Estimated transaction cost per position change (basis points)", 0, 50, 10, 1,
                            help="10 basis points = 0.10%. This is a simplified cost assumption, not your exact brokerage/taxes.")
        starting_cash = st.number_input(f"Starting capital ({RUPEE})", min_value=1000, max_value=100000000, value=100000, step=10000)

        bt = df.copy().dropna(subset=["Close"])
        close = bt["Close"]
        if strategy == "EMA crossover (20/50)":
            raw = (bt["EMA 20"] > bt["EMA 50"]).astype(float)
        elif strategy == "RSI mean reversion (30/70)":
            # Hold after RSI recovers above 30; exit when RSI reaches 70 or falls below 30.
            raw = pd.Series(np.nan, index=bt.index, dtype=float)
            state = 0.0
            for idx, value in bt["RSI 14"].items():
                if pd.notna(value):
                    if value < 30: state = 1.0
                    elif value > 70: state = 0.0
                raw.loc[idx] = state
            raw = raw.ffill().fillna(0.0)
        elif strategy == "MACD crossover":
            raw = (bt["MACD"] > bt["MACD Signal"]).astype(float)
        else:
            raw = ((bt["EMA 20"] > bt["EMA 50"]) & (bt["MACD"] > bt["MACD Signal"]) & (bt["RSI 14"] > 50)).astype(float)
        if direction == "Long / short (simplified)":
            # Explicitly simplified: short when long signal is off.
            raw = raw * 2 - 1
        # Apply signal on next bar, avoiding same-day close look-ahead.
        position = raw.shift(1).fillna(0.0)
        daily_returns = close.pct_change().fillna(0.0)
        turnover = position.diff().abs().fillna(position.abs())
        cost = turnover * (fee_bps / 10000.0)
        strategy_returns = position * daily_returns - cost
        equity = starting_cash * (1 + strategy_returns).cumprod()
        benchmark = starting_cash * (1 + daily_returns).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1
        total_return = equity.iloc[-1] / starting_cash - 1
        years = max((bt.index[-1] - bt.index[0]).days / 365.25, 1/365.25)
        cagr = (equity.iloc[-1] / starting_cash) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1
        daily_std = strategy_returns.std()
        sharpe = (strategy_returns.mean() / daily_std * np.sqrt(252)) if daily_std and pd.notna(daily_std) else np.nan
        changes = position.diff().fillna(position.iloc[0]).abs()
        trades = int((changes > 0).sum())
        closed_trade_returns = daily_returns[(position.shift(1).fillna(0) != 0) & (position == 0)]
        win_rate = float((closed_trade_returns > 0).mean() * 100) if len(closed_trade_returns) else np.nan
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Strategy ending value", money(float(equity.iloc[-1])), f"{total_return*100:+.2f}% total")
        b2.metric("Benchmark ending value", money(float(benchmark.iloc[-1])), f"{(benchmark.iloc[-1]/starting_cash-1)*100:+.2f}% total")
        b3.metric("Max drawdown", f"{drawdown.min()*100:.2f}%")
        b4.metric("CAGR (approx.)", f"{cagr*100:.2f}%")
        b5,b6,b7 = st.columns(3)
        b5.metric("Sharpe ratio (approx.)", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
        b6.metric("Position changes", f"{trades}")
        b7.metric("Win rate (simple exit-day proxy)", f"{win_rate:.1f}%" if pd.notna(win_rate) else "N/A")
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy equity"))
        chart.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Market benchmark"))
        chart.update_layout(title="Equity curve", xaxis_title="Date", yaxis_title=f"Portfolio value ({RUPEE})", template="plotly_white", height=430)
        st.plotly_chart(chart, use_container_width=True)
        ddfig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown*100, name="Drawdown %", fill="tozeroy"))
        ddfig.update_layout(title="Strategy drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_white", height=300)
        st.plotly_chart(ddfig, use_container_width=True)
        with st.expander("View latest backtest rows"):
            detail = pd.DataFrame({"Close": close, "Signal before shift": raw, "Position used": position,
                                   "Daily return %": daily_returns*100, "Strategy return %": strategy_returns*100,
                                   "Strategy equity": equity, "Benchmark equity": benchmark, "Drawdown %": drawdown*100})
            st.dataframe(detail.tail(100).round(3), use_container_width=True)
        st.warning("Backtest results are historical simulations, not predictions. They exclude slippage, taxes, liquidity limits, corporate-action complexities and exact brokerage schedules. Long/short mode is simplified and does not model margin or short-selling constraints. Try multiple periods and stocks; avoid selecting a strategy only because it performed well in the past.")



