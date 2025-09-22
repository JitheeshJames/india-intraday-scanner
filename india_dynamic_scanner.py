#!/usr/bin/env python3
# coding: utf-8

"""
India NSE50 Intraday Scanner - Updated
- Config-driven NSE50 universe
- VWAP + Opening Range breakout
- Relative Strength vs index
- Volume impulse check
- Lot-based position sizing
- Telegram messaging
- Dry-run / test mode
"""

import os, sys, json, time, math, logging, argparse
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
import pytz
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

# ---------- Paths & Logging ----------
IST = pytz.timezone("Asia/Kolkata")
ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)]
)
log = logging.getLogger("intraday_scanner")

# ---------- Config ----------
CONFIG_FILE = ROOT / "config.json"

DEFAULT_CONFIG = {
    "india": {
        "dynamic_universe_size": 50,
        "risk_pct": 0.5,
        "buffer_bp": 5,
        "min_rs_threshold": 0.0,
        "min_volume_impulse": 1.0,
        "index": "^NSEI",
        "require_both_or_vwap": False,
        "rs_weight": 1.0,
        "vol_weight": 1.0,
        "holidays": []
    },
    "capital_inr": 100000,
    "lots": {"DEFAULT": 1},
    "min_stop_distance": 0.0025,
    "data": {
        "primary": "yfinance",
        "yf_batch_size": 8,
        "yf_retries": 3,
        "yf_retry_backoff": 2
    }
}

def load_cfg(path=CONFIG_FILE):
    if path.exists():
        try:
            with open(path) as f:
                cfg = json.load(f)
            c = DEFAULT_CONFIG.copy()
            c.update(cfg)
            if "india" in cfg:
                c["india"].update(cfg.get("india", {}))
            return c
        except Exception as e:
            log.warning("Failed to load config.json: %s, using defaults", e)
    return DEFAULT_CONFIG.copy()

def get_lot_size(symbol, lots_map):
    return lots_map.get(symbol.upper(), lots_map.get("DEFAULT", 1))

# ---------- Utilities ----------
def retry_fn(fn, retries=3, backoff=1, allowed_exceptions=(Exception,)):
    last_exc = None
    for i in range(retries):
        try:
            return fn()
        except allowed_exceptions as e:
            last_exc = e
            log.warning("Attempt %d/%d failed: %s", i+1, retries, e)
            time.sleep(backoff*(2**i))
    raise last_exc

def ensure_index_tz(df, tz=IST):
    if df is None or df.empty:
        return df
    idx = df.index
    if getattr(idx, "tz", None) is None:
        try:
            df = df.tz_localize("UTC").tz_convert(tz)
        except Exception:
            df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)
    return df

# ---------- Data Fetching ----------
def yf_download_tickers(tickers, period="1d", interval="1m", batch_size=8, retries=3, backoff=2):
    if yf is None:
        raise RuntimeError("yfinance not installed")
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        def _download():
            log.info("Downloading batch %d-%d: %s", i, i+len(batch)-1, batch)
            return yf.download(tickers=batch, period=period, interval=interval, group_by="ticker", progress=False)
        try:
            data = retry_fn(_download, retries=retries, backoff=backoff)
        except Exception as e:
            log.error("Batch download failed: %s", e)
            continue
        # Normalize
        if len(batch)==1:
            all_data[batch[0]] = data
        else:
            for t in batch:
                try:
                    all_data[t] = data[t]
                except Exception:
                    continue
    return all_data

# ---------- Indicators ----------
def compute_vwap(df):
    if df is None or df.empty or not {"High","Low","Close","Volume"}.issubset(df.columns):
        return pd.Series(dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"])/3
    pv = tp*df["Volume"]
    return pv.cumsum()/df["Volume"].cumsum()

def compute_opening_range(df, open_time=dt_time(9,15), window_minutes=30):
    df = ensure_index_tz(df)
    if df.empty:
        return None, None, df
    last_day = df.index[-1].date()
    market_open = IST.localize(datetime.combine(last_day, open_time))
    window_end = market_open + timedelta(minutes=window_minutes)
    mask = (df.index >= market_open) & (df.index < window_end)
    or_df = df.loc[mask]
    if or_df.empty:
        return None, None, or_df
    return float(or_df["High"].max()), float(or_df["Low"].min()), or_df

def analyze_volume_impulse(df, lookback_recent=4):
    if df is None or df.empty or "Volume" not in df.columns:
        return 1.0
    overall = df["Volume"].mean()
    recent = df["Volume"].iloc[-lookback_recent:].mean() if len(df)>=lookback_recent else df["Volume"].iloc[-1]
    return float(recent/overall) if overall>0 else 1.0

def rs_vs_index(stock_df, index_df):
    try:
        s_ret = stock_df["Close"].iloc[-1]/stock_df["Close"].iloc[0]-1
        i_ret = index_df["Close"].iloc[-1]/index_df["Close"].iloc[0]-1
        return float(s_ret-i_ret)
    except Exception:
        return 0.0

def analyze_price_action(df, cfg_india):
    if df is None or df.empty:
        return {"bias":"NEUTRAL","entry":0.0,"stop":0.0}
    df = ensure_index_tz(df)
    latest = df["Close"].iloc[-1]
    vwap_ser = compute_vwap(df)
    latest_vwap = float(vwap_ser.iloc[-1]) if not vwap_ser.empty else latest
    orh, orl, _ = compute_opening_range(df)
    require_both = cfg_india.get("require_both_or_vwap", False)
    long_cond = latest>orh if orh else False
    short_cond = latest<orl if orl else False
    vwap_long = latest>latest_vwap
    vwap_short = not vwap_long
    if require_both:
        is_long = long_cond and vwap_long
        is_short = short_cond and vwap_short
    else:
        is_long = long_cond or vwap_long
        is_short = short_cond or vwap_short
    if is_long:
        return {"bias":"LONG","entry":orh or latest,"stop":orl or df["Low"].min()}
    elif is_short:
        return {"bias":"SHORT","entry":orl or latest,"stop":orh or df["High"].max()}
    else:
        return {"bias":"NEUTRAL","entry":latest,"stop":latest}

# ---------- Position Sizing ----------
def build_levels(p, cfg, capital):
    buffer_bp = cfg["india"].get("buffer_bp",5)
    risk_pct = cfg["india"].get("risk_pct",0.5)
    min_stop = cfg.get("min_stop_distance",0.5)
    lots_map = cfg.get("lots",{"DEFAULT":1})
    bias = p["bias"]
    entry, stop = p["entry"], p["stop"]
    lot_size = get_lot_size(p["ticker"], lots_map)
    if bias=="LONG":
        entry = entry*(1+buffer_bp/10000)
        stop = stop*(1-buffer_bp/10000)
        risk = max(entry-stop,min_stop)
    elif bias=="SHORT":
        entry = entry*(1-buffer_bp/10000)
        stop = stop*(1+buffer_bp/10000)
        risk = max(stop-entry,min_stop)
    else:
        return {"entry":0,"stop":0,"target":0,"qty":0,"risk":0}
    risk_amount = capital*(risk_pct/100)
    raw_qty = risk_amount/risk if risk>0 else 0
    qty_shares = int(math.floor(raw_qty))
    qty_final = (qty_shares//lot_size)*lot_size if qty_shares>=lot_size else 0
    target = entry+2*risk if bias=="LONG" else entry-2*risk
    return {"entry":round(entry,2),"stop":round(stop,2),"target":round(target,2),"qty":qty_final,"risk":round(risk,4)}

# ---------- Telegram ----------
def send_tg(msg):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("Telegram credentials not set.")
        return False
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id":chat_id,"text":msg,"parse_mode":"HTML","disable_web_page_preview":True},timeout=10)
        if r.status_code==200:
            log.info("Telegram message sent")
            return True
        else:
            log.error("Telegram send failed: %s",r.text)
            return False
    except Exception as e:
        log.exception("Telegram send error: %s", e)
        return False

# ---------- Universe & Scan ----------
def fetch_universe(cfg):
    # NSE50 default list
    base = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN","BAJFINANCE","BHARTIARTL",
        "KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN","SUNPHARMA","TATAMOTORS","WIPRO","ULTRACEMCO",
        "ADANIENT","JSWSTEEL","TATASTEEL","HCLTECH","NTPC","POWERGRID","BAJAJFINSV","ONGC","HDFC","GRASIM",
        "SBILIFE","COALINDIA","BPCL","EICHERMOT","DRREDDY","HDFCLIFE","DIVISLAB","UPL","BRITANNIA","TATACONSUM",
        "TECHM","SHREECEM","M&M","CIPLA","ADANIPORTS","TATAPOWER","IOC","VEDL","INDUSINDBK","APOLLOHOSP"
    ]
    size = cfg["india"].get("dynamic_universe_size",50)
    tickers = [f"{s}.NS" for s in base[:size]]
    return tickers

def pick_signals(cfg):
    ist_now = datetime.now(IST)
    today = ist_now.date()
    # Weekend/Holiday skip
    if today.weekday()>=5 or today.strftime("%Y-%m-%d") in cfg["india"].get("holidays",[]):
        log.info("Market closed today")
        return pd.DataFrame(), [], "Closed"
    universe = fetch_universe(cfg)
    log.info("Universe size: %d", len(universe))
    # Fetch index
    try:
        index_df = yf.download(cfg["india"].get("index","^NSEI"),period="1d",interval="1m",progress=False)
        index_df = ensure_index_tz(index_df)
    except Exception as e:
        log.error("Index fetch failed: %s", e)
        return pd.DataFrame(), [], "Index error"
    # Fetch stocks
    data = yf_download_tickers(universe, period="1d", interval="1m",
                               batch_size=cfg["data"].get("yf_batch_size",8),
                               retries=cfg["data"].get("yf_retries",3),
                               backoff=cfg["data"].get("yf_retry_backoff",2))
    candidates = []
    for ticker, df in data.items():
        try:
            if df.empty or len(df)<3: continue
            df = ensure_index_tz(df)
            rs = rs_vs_index(df, index_df)
            vol_ratio = analyze_volume_impulse(df)
            analysis = analyze_price_action(df,cfg["india"])
            bias = analysis["bias"]
            if bias=="NEUTRAL": continue
            # RS filter
            if bias=="LONG" and rs<cfg["india"].get("min_rs_threshold",0.0): continue
            if bias=="SHORT" and rs>-cfg["india"].get("min_rs_threshold",0.0): continue
            candidates.append({"ticker":ticker.replace(".NS",""),"bias":bias,"rs":rs,"vol_ratio":vol_ratio,
                               "entry":analysis["entry"],"stop":analysis["stop"]})
        except Exception:
            log.exception("Error processing %s", ticker)
            continue
    if not candidates:
        return pd.DataFrame(), [], "No candidates"
    dfc = pd.DataFrame(candidates)
    # Scoring
    rs_w = cfg["india"].get("rs_weight",1.0)
    vol_w = cfg["india"].get("vol_weight",1.0)
    dfc["score"] = (dfc["rs"].abs()*100*rs_w)+(dfc["vol_ratio"]*10*vol_w)
    dfc = dfc.sort_values("score",ascending=False)
    picks = dfc.head(5).to_dict("records")
    return dfc, picks, f"Scan @ {ist_now.strftime('%H:%M')}"

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    cfg = load_cfg()
    capital = cfg.get("capital_inr",100000)
    dfc, picks, desc = pick_signals(cfg)
    if not picks:
        msg = f"No clean setups - {desc}"
        log.info(msg)
        if not args.test: send_tg(msg)
        return
    lines = [f"🎯 Top picks - {datetime.now(IST).strftime('%d-%b-%Y %H:%M')} - {desc}"]
    for p in picks:
        lvl = build_levels(p,cfg,capital)
        if lvl.get("qty",0)<=0:
            lines.append(f"{p['ticker']} {p['bias']} - Entry {lvl.get('entry')} Stop {lvl.get('stop')} - Qty insufficient")
            continue
        rr = abs((lvl['target']-lvl['entry'])/max(abs(lvl['entry']-lvl['stop']),1e-6))
        lines.append(f"{p['ticker']} <b>{p['bias']}</b> Entry:{lvl['entry']} Stop:{lvl['stop']} Target:{lvl['target']} Qty:{lvl['qty']} R:R≈{rr:.2f} RS:{p['rs']:.3f} Vol:{p['vol_ratio']:.2f}x")
    msg = "\n".join(lines)
    if args.test:
        print(msg)
    else:
        send_tg(msg)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log.exception("Fatal scanner error: %s", e)
        try: send_tg(f"⚠️ Scanner Error: {e}")
        except Exception: pass
        sys.exit(1)
