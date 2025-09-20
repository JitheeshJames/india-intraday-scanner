import os, math, pytz, datetime as dt, time
import yaml, json, requests
from bs4 import BeautifulSoup
import numpy as np, pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from telegram import Bot

IST = pytz.timezone("Asia/Kolkata")

def load_cfg():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def is_trading_day(date_ist):
    try:
        nse = mcal.get_calendar("XNSE")
        sched = nse.schedule(start_date=date_ist, end_date=date_ist)
        return not sched.empty
    except Exception:
        return date_ist.weekday() < 5  # Mon-Fri

def fetch_active_stocks(limit=100):
    """Get the most active stocks by volume from NSE"""
    
    # Method 1: Direct from NSE volume data - sometimes blocked
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get('data', []))
            if not df.empty and 'symbol' in df.columns:
                return df['symbol'].tolist()[:limit]
    except Exception as e:
        print(f"NSE API fetch failed: {e}")
    
    # Method 2: F&O stocks as fallback (reliable universe of liquid names)
    try:
        url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
        df = pd.read_csv(url)
        if 'SYMBOL' in df.columns:
            return df['SYMBOL'].tolist()[:limit]
    except Exception as e:
        print(f"NSE F&O list fetch failed: {e}")
    
    # Method 3: Hardcoded fallback of Nifty 100 components
    nifty100 = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HDFC", "ITC", 
                "KOTAKBANK", "LT", "HINDUNILVR", "SBIN", "BAJFINANCE", "AXISBANK", 
                "BHARTIARTL", "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "BAJAJFINSV",
                "TATAMOTORS", "ADANIENT", "NTPC", "POWERGRID", "ULTRACEMCO", "HCLTECH",
                "M&M", "JSWSTEEL", "TATASTEEL", "DMART", "NESTLEIND", "TECHM", "ADANIPORTS", 
                "WIPRO", "ONGC", "SBILIFE", "GRASIM", "COALINDIA", "INDUSINDBK", "DIVISLAB", 
                "CIPLA", "HDFCLIFE", "BAJAJ-AUTO", "BPCL", "ADANIPOWER", "TATAPOWER", 
                "HINDALCO", "IOC", "EICHERMOT", "ZOMATO", "GODREJCP"]
    
    return [s + ".NS" for s in nifty100[:limit]]

def yf_intraday(tickers, start_ist, end_ist):
    """Fetch 1-minute data for the given tickers"""
    start_utc = start_ist.astimezone(pytz.UTC)
    end_utc = end_ist.astimezone(pytz.UTC)
    
    # Split into batches to avoid request timeouts (25 tickers per batch)
    all_data = {}
    batch_size = 25
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(
                tickers=batch, interval="1m",
                start=start_utc, end=end_utc,
                auto_adjust=True, progress=False,
                group_by="ticker", prepost=False
            )
            
            # Process batch results
            if len(batch) == 1:
                # Handle single ticker case
                all_data[batch[0]] = df
            else:
                # Handle multi-ticker case
                for ticker in batch:
                    if ticker in df:
                        all_data[ticker] = df[ticker].copy()
            
            # Avoid rate limiting
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching batch {i}-{i+batch_size}: {e}")
    
    return all_data

def vwap(df_1m):
    tp = (df_1m["High"] + df_1m["Low"] + df_1m["Close"]) / 3.0
    v = df_1m["Volume"].replace(0, np.nan).fillna(0)
    return (tp.mul(v).cumsum() / v.cumsum().replace(0, np.nan)).ffill()

def opening_range(df_1m):
    df = df_1m.copy().tz_localize("UTC").tz_convert(IST)
    mask = (df.index.time >= dt.time(9,15)) & (df.index.time < dt.time(9,30))
    window = df.loc[mask]
    if window.empty:
        return np.nan, np.nan
    return float(window["High"].max()), float(window["Low"].min())

def last_bar_at(df_1m, t_ist):
    df = df_1m.copy().tz_localize("UTC").tz_convert(IST)
    df = df[df.index <= t_ist]
    return df.iloc[-1] if len(df) else None

def rs_vs_index(stock_df, index_df):
    """Calculate relative strength vs index from 09:15 to 09:45"""
    today = dt.datetime.now(IST).date()
    t0_ist = IST.localize(dt.datetime.combine(today, dt.time(9,15)))
    t1_ist = IST.localize(dt.datetime.combine(today, dt.time(9,45)))
    
    s = stock_df.tz_localize("UTC").tz_convert(IST).loc[t0_ist:t1_ist]["Close"]
    i = index_df.tz_localize("UTC").tz_convert(IST).loc[t0_ist:t1_ist]["Close"]
    
    if len(s) < 2 or len(i) < 2:
        return 0.0
    
    return (s[-1]/s[0] - 1) - (i[-1]/i[0] - 1)

def pick_signals(cfg):
    today = dt.datetime.now(IST).date()
    if not is_trading_day(today):
        return [], []

    # Get active stocks for today
    dynamic_limit = cfg["india"]["dynamic_universe_size"]
    active_stocks = fetch_active_stocks(limit=dynamic_limit)
    
    # Define time window
    start = IST.localize(dt.datetime.combine(today, dt.time(9,0)))
    bias_cut = IST.localize(dt.datetime.combine(today, dt.time(9,45)))
    
    # Get index data first
    idx = cfg["india"]["index"]
    idx_data = yf.download(idx, interval="1m", 
                          start=start.astimezone(pytz.UTC), 
                          end=bias_cut.astimezone(pytz.UTC),
                          auto_adjust=True, progress=False)
    
    if idx_data.empty:
        return [], []
    
    # Get index setup parameters
    idx_orh, idx_orl = opening_range(idx_data)
    idx_vwap_last = vwap(idx_data).iloc[-1]
    idx_last = last_bar_at(idx_data, bias_cut)
    
    if idx_last is None:
        return [], []
    
    idx_up = (idx_last["Close"] > idx_orh) and (idx_last["Close"] > idx_vwap_last)
    idx_dn = (idx_last["Close"] < idx_orl) and (idx_last["Close"] < idx_vwap_last)

    # Fetch data for all active stocks
    stock_data = yf_intraday(active_stocks, start, bias_cut)
    
    rows = []
    for ticker, df in stock_data.items():
        if df.empty or 'Close' not in df.columns:
            continue
            
        try:
            orh, orl = opening_range(df)
            if np.isnan(orh) or np.isnan(orl):
                continue
                
            vwap_values = vwap(df)
            if vwap_values.empty:
                continue
                
            vwap_last = vwap_values.iloc[-1]
            last = last_bar_at(df, bias_cut)
            
            if last is None:
                continue
            
            # Calculate relative strength vs index
            rs = rs_vs_index(df[["Close"]], idx_data[["Close"]])
            
            # Volume impulse
            try:
                avg_day_vol = yf.download(ticker, period="30d", interval="1d", 
                                        auto_adjust=True, progress=False)["Volume"].mean()
                vol30m = float(df["Volume"].sum())
                vol_ratio = (vol30m / max(avg_day_vol, 1)) if avg_day_vol else 0.0
            except:
                vol_ratio = 0.0
            
            # Determine bias
            long_ok = (last["Close"] > orh) and (last["Close"] > vwap_last) and idx_up and (rs > 0)
            short_ok = (last["Close"] < orl) and (last["Close"] < vwap_last) and idx_dn and (rs < 0)
            
            bias = "LONG" if long_ok else ("SHORT" if short_ok else "NONE")
            
            rows.append({
                "ticker": ticker, "orh": orh, "orl": orl, "last": float(last["Close"]),
                "vwap": float(vwap_last), "rs": float(rs), "vol_ratio": float(vol_ratio),
                "bias": bias
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    dfc = pd.DataFrame(rows)
    picks = dfc[dfc["bias"] != "NONE"].copy()
    
    if len(picks) == 0:
        return dfc, []
    
    # Sort by combined score (RS importance × 100 + volume ratio × 10)
    picks["score"] = picks["rs"].abs() * 100 + picks["vol_ratio"] * 10
    picks = picks.sort_values("score", ascending=False).head(cfg["india"]["top_n"]).to_dict(orient="records")
    
    return dfc, picks

def build_levels(p, risk_pct, capital, buffer_bp):
    if p["bias"] == "LONG":
        entry = p["orh"] * (1 + buffer_bp/10000.0)
        stop = min(p["orl"], p["vwap"])
        risk = max(entry - stop, 0.01)
        qty = max(0, math.floor((capital * risk_pct) / risk))
        tgt = entry + 2 * risk
    elif p["bias"] == "SHORT":
        entry = p["orl"] * (1 - buffer_bp/10000.0)
        stop = max(p["orh"], p["vwap"])
        risk = max(stop - entry, 0.01)
        qty = max(0, math.floor((capital * risk_pct) / risk))
        tgt = entry - 2 * risk
    else:
        entry = stop = tgt = 0.0
        qty = 0
    
    return entry, stop, tgt, qty

def send_tg(msg):
    Bot(token=os.environ["TELEGRAM_TOKEN"]).send_message(
        chat_id=os.environ["TELEGRAM_CHAT_ID"], 
        text=msg, 
        parse_mode="HTML", 
        disable_web_page_preview=True
    )

def main():
    cfg = load_cfg()
    
    if not is_trading_day(dt.datetime.now(IST).date()):
        send_tg("NSE Holiday/Closed: Scanner skipped.")
        return
    
    # Let user know scan started
    dynamic_limit = cfg["india"]["dynamic_universe_size"]
    send_tg(f"🔍 Scanning top {dynamic_limit} active NSE stocks...")
    
    # Run the scan
    table, picks = pick_signals(cfg)
    
    if not picks:
        send_tg("No clean intraday setups found by 09:45 IST (ORB+VWAP+Index filters).")
        return
    
    risk_pct = cfg["india"]["risk_pct"]
    capital = cfg.get("capital_inr", 100000)
    buf = cfg["india"]["buffer_bp"]
    
    lines = [f"🎯 Top intraday picks ({len(picks)}) - {dt.datetime.now(IST).strftime('%d-%b-%Y')}:"]
    
    for p in picks:
        entry, stop, tgt, qty = build_levels(p, risk_pct, capital, buf)
        rr = abs((tgt-entry)/max(entry-stop, 1e-6)) if p["bias"]=="LONG" else abs((entry-tgt)/max(stop-entry, 1e-6))
        
        lines.append(
            f"{p['ticker']}  <b>{p['bias']}</b>"
            f"  Entry: {entry:.2f} | Stop: {stop:.2f} | Target: {tgt:.2f} | Qty: {qty}"
            f"  RS: {p['rs']:.3f} | VolImpulse: {p['vol_ratio']:.3f} | R:R≈{rr:.1f}x"
        )
    
    send_tg("
".join(lines))

if __name__ == "__main__":
    main()
