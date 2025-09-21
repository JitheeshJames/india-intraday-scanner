#!/usr/bin/env python3

-- coding: utf-8 --

""" Refactored India Intraday Scanner

Improved robustness for yfinance failures (retries + fallback hooks)

Explicit timezone handling (IST)

Opening Range computed from completed candles with clear inclusive/exclusive rules

VWAP, volume impulse, RS calculation improved and configurable

Position sizing with min_stop_distance and lot-size rounding (simple mapping)

Config-driven scores & weights

Better logging, Telegram formatting, and safe error handling

Dry-run / test mode


Note: This file is intended to replace your existing scanner. Update config.json or pass environment variables as needed. """

import os import sys import json import time as time_mod import argparse import logging import math from datetime import datetime, date, time, timedelta from pathlib import Path

import pytz import numpy as np import pandas as pd

Optional dependency: yfinance

try: import yfinance as yf except Exception: yf = None

---------- Configuration & logging ----------

IST = pytz.timezone("Asia/Kolkata") ROOT = Path(file).parent LOG_DIR = ROOT / "logs" LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log" logging.basicConfig( level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", handlers=[ logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE) ] ) log = logging.getLogger("intraday_scanner")

DEFAULT_CONFIG = { "india": { "dynamic_universe_size": 100, "risk_pct": 0.5, "buffer_bp": 5, "min_rs_threshold": 0.0, "min_volume_impulse": 1.0, "index": "^NSEI", "rs_weight": 1.0, "vol_weight": 1.0, "require_both_or_vwap": False  # allow OR or VWAP (more permissive) }, "capital_inr": 100000, "data": { "primary": "yfinance", "yf_batch_size": 8, "yf_retries": 3, "yf_retry_backoff": 2 }, "lots": {"DEFAULT": 1}, "min_stop_distance": 0.5 }

CONFIG_FILE = ROOT / "config.json"

def load_cfg(path=CONFIG_FILE): if path.exists(): try: with open(path, "r") as f: cfg = json.load(f) # merge defaults (simple shallow merge) c = DEFAULT_CONFIG.copy() c.update(cfg) if "india" in cfg: c["india"].update(cfg.get("india", {})) return c except Exception as e: log.warning("Failed to load config.json: %s - using defaults", e) return DEFAULT_CONFIG.copy()

---------- Helpers: retries, tz, lot sizing ----------

def retry_fn(fn, retries=3, backoff=1, allowed_exceptions=(Exception,)): last_exc = None for i in range(retries): try: return fn() except allowed_exceptions as e: last_exc = e log.warning("Attempt %d/%d failed: %s", i + 1, retries, e) time_mod.sleep(backoff * (2 ** i)) raise last_exc

def ensure_index_tz(df, tz=IST): """Ensure df.index is timezone-aware and converted to IST.""" if df is None or df.empty: return df idx = df.index if getattr(idx, "tz", None) is None: # assume UTC if naive (yfinance often returns UTC) try: df = df.tz_localize("UTC").tz_convert(tz) except Exception: # last resort: localize to IST (if data already was IST) df = df.tz_localize(tz) else: df = df.tz_convert(tz) return df

def get_lot_size(symbol, lots_map): symbol = symbol.upper() return lots_map.get(symbol, lots_map.get("DEFAULT", 1))

---------- Data fetching ----------

def yf_download_tickers(tickers, period="1d", interval="1m", batch_size=8, retries=3, backoff=2): if yf is None: raise RuntimeError("yfinance not available. Install yfinance or configure a different data provider.")

all_data = {}
for i in range(0, len(tickers), batch_size):
    batch = tickers[i : i + batch_size]

    def _download():
        log.info("Downloading batch %d-%d: %s", i, i + len(batch) - 1, batch)
        return yf.download(tickers=batch, period=period, interval=interval, group_by="ticker", progress=False, auto_adjust=False)

    try:
        data = retry_fn(_download, retries=retries, backoff=backoff)
    except Exception as e:
        log.error("Batch download failed after retries: %s", e)
        # skip this batch but continue
        continue

    # normalize result
    if len(batch) == 1:
        ticker = batch[0]
        if not data.empty:
            df = data.copy()
            all_data[ticker] = df
    else:
        # multi-index columns
        for ticker in batch:
            try:
                if ticker in data.columns.levels[0]:
                    df = data[ticker].copy()
                    if not df.empty:
                        all_data[ticker] = df
            except Exception:
                # older versions of yfinance might have different shapes
                pass

return all_data

---------- Indicators & logic ----------

def compute_vwap(df): if df is None or df.empty: return pd.Series(dtype=float) if not {"High", "Low", "Close", "Volume"}.issubset(df.columns): return pd.Series(dtype=float) tp = (df["High"] + df["Low"] + df["Close"]) / 3.0 pv = (tp * df["Volume"])  # per-bar PV return pv.cumsum() / df["Volume"].cumsum()

def compute_opening_range(df, open_time=time(9, 15), window_minutes=30): """Return OR high, low, and mask for OR bars. Uses completed candles: includes bars where index >= market_open and index < market_open+window. """ if df is None or df.empty: return None, None, pd.DataFrame()

df = ensure_index_tz(df)
last_day = df.index[-1].date()
market_open_dt = IST.localize(datetime.combine(last_day, open_time))
window_end = market_open_dt + timedelta(minutes=window_minutes)
mask = (df.index >= market_open_dt) & (df.index < window_end)
window_df = df.loc[mask]
if window_df.empty:
    return None, None, window_df
return float(window_df["High"].max()), float(window_df["Low"].min()), window_df

def analyze_volume_impulse(df, lookback_recent=4): if df is None or df.empty or "Volume" not in df.columns: return 1.0 overall = df["Volume"].mean() recent = df["Volume"].iloc[-lookback_recent:].mean() if len(df) >= lookback_recent else df["Volume"].iloc[-1] if overall == 0 or np.isnan(overall) or overall == 0.0: return 1.0 return float(recent / overall)

def rs_vs_index(stock_df, index_df): if stock_df is None or index_df is None or stock_df.empty or index_df.empty: return 0.0 try: s_first = float(stock_df["Close"].iloc[0]) s_last = float(stock_df["Close"].iloc[-1]) i_first = float(index_df["Close"].iloc[0]) i_last = float(index_df["Close"].iloc[-1]) s_ret = s_last / s_first - 1.0 i_ret = i_last / i_first - 1.0 return float(s_ret - i_ret) except Exception: return 0.0

def analyze_price_action(df, cfg_india): # returns bias, entry, stop if df is None or df.empty: return {"bias": "NEUTRAL", "entry": 0.0, "stop": 0.0}

df = ensure_index_tz(df)
closes = df["Close"]
if len(closes) < 2:
    return {"bias": "NEUTRAL", "entry": 0.0, "stop": 0.0}

latest_close = float(closes.iloc[-1])
prev_close = float(closes.iloc[-2])

vwap_ser = compute_vwap(df)
latest_vwap = float(vwap_ser.iloc[-1]) if not vwap_ser.empty else latest_close

orh, orl, or_df = compute_opening_range(df)

require_both = cfg_india.get("require_both_or_vwap", False)

# Logic: prefer OR breakout, but allow OR or VWAP depending on config
long_cond = False
short_cond = False

if orh is not None and orl is not None:
    if latest_close > orh:
        long_cond = True
    if latest_close < orl:
        short_cond = True

# VWAP confirmation
if latest_close > latest_vwap:
    vwap_long = True
else:
    vwap_long = False
vwap_short = not vwap_long

# Combine per config
if require_both:
    is_long = long_cond and vwap_long
    is_short = short_cond and vwap_short
else:
    is_long = long_cond or vwap_long
    is_short = short_cond or vwap_short

if is_long:
    bias = "LONG"
    entry = orh if orh is not None else latest_close
    stop = orl if orl is not None else float(df["Low"].min())
elif is_short:
    bias = "SHORT"
    entry = orl if orl is not None else latest_close
    stop = orh if orh is not None else float(df["High"].max())
else:
    bias = "NEUTRAL"
    entry = stop = latest_close

return {"bias": bias, "entry": float(entry), "stop": float(stop)}

def build_levels(p, cfg, capital): buffer_bp = cfg["india"].get("buffer_bp", 5) risk_pct = cfg["india"].get("risk_pct", 0.5) min_stop_distance = cfg.get("min_stop_distance", 0.5) lots_map = cfg.get("lots", {"DEFAULT": 1})

bias = p.get("bias", "NEUTRAL")
entry_raw = float(p.get("entry", 0.0))
stop_raw = float(p.get("stop", 0.0))
symbol = p.get("ticker", "UNKNOWN")

lot_size = get_lot_size(symbol, lots_map)

if bias == "LONG":
    entry = entry_raw * (1 + buffer_bp / 10000.0)
    stop = stop_raw * (1 - buffer_bp / 10000.0)
    risk = max(entry - stop, min_stop_distance)
elif bias == "SHORT":
    entry = entry_raw * (1 - buffer_bp / 10000.0)
    stop = stop_raw * (1 + buffer_bp / 10000.0)
    risk = max(stop - entry, min_stop_distance)
else:
    return {"entry": 0.0, "stop": 0.0, "target": 0.0, "qty": 0}

# Position sizing
risk_amount = capital * (risk_pct / 100.0)
raw_qty = risk_amount / risk if risk > 0 else 0
qty_shares = int(math.floor(raw_qty))

# round down to nearest lot
if qty_shares < lot_size:
    qty_final = 0
else:
    qty_final = (qty_shares // lot_size) * lot_size

target = entry + (2 * risk) if bias == "LONG" else entry - (2 * risk)

return {
    "entry": round(entry, 2),
    "stop": round(stop, 2),
    "target": round(target, 2),
    "qty": int(qty_final),
    "risk": round(risk, 4),
    "raw_qty": raw_qty
}

---------- Telegram ----------

def send_tg(msg): token = os.environ.get("TELEGRAM_TOKEN") chat_id = os.environ.get("TELEGRAM_CHAT_ID") if not token or not chat_id: log.warning("Telegram credentials not set. Skipping send.") return False

url = f"https://api.telegram.org/bot{token}/sendMessage"
payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}

try:
    import requests
    r = requests.post(url, json=payload, timeout=10)
    if r.status_code == 200:
        log.info("Telegram message sent")
        return True
    else:
        log.error("Telegram send failed: %s", r.text)
        return False
except Exception as e:
    log.exception("Error sending Telegram: %s", e)
    return False

---------- Main scanning logic ----------

def fetch_universe(cfg): # Keep same method: static NIFTY components + configurable size # You may replace this with a dynamic liquidity scrape or an API base = [ "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "TATAMOTORS", "WIPRO", "ULTRACEMCO", "ADANIENT", "JSWSTEEL", "TATASTEEL", "HCLTECH", "NTPC" ] size = cfg["india"].get("dynamic_universe_size", 100) tickers = [f"{s}.NS" for s in base[:max(5, size)]] return tickers

def pick_signals(cfg, hour_of_day=None, minute_of_hour=None): ist_now = datetime.now(IST) today = ist_now.date()

if hour_of_day is None:
    hour_of_day = ist_now.hour
if minute_of_hour is None:
    minute_of_hour = ist_now.minute

scan_time = IST.localize(datetime.combine(today, time(hour_of_day, minute_of_hour)))
log.info("Running scan for %s", scan_time)

# Simple trading day check: weekend + optional holidays maintenance (users should keep config updated)
if scan_time.weekday() >= 5:
    log.info("Weekend - skipping scan")
    return pd.DataFrame(), [], "Weekend"

# Determine intervals
if hour_of_day == 9:
    interval = "1m"
    period = "1d"
else:
    interval = "5m"
    period = "1d"

universe = fetch_universe(cfg)
log.info("Universe size: %d", len(universe))

# fetch index first
index_symbol = cfg["india"].get("index", "^NSEI")
try:
    index_df = yf.download(index_symbol, period=period, interval=interval, progress=False)
    index_df = ensure_index_tz(index_df)
except Exception as e:
    log.error("Failed to fetch index data: %s", e)
    return pd.DataFrame(), [], "Index data error"

# fetch stocks
data = yf_download_tickers(universe, period=period, interval=interval, batch_size=cfg["data"].get("yf_batch_size", 8),
                           retries=cfg["data"].get("yf_retries", 3), backoff=cfg["data"].get("yf_retry_backoff", 2))

candidates = []
for ticker, df in data.items():
    try:
        if df.empty or len(df) < 3:
            continue
        df = ensure_index_tz(df)
        rs = rs_vs_index(df, index_df)
        vol_ratio = analyze_volume_impulse(df)
        analysis = analyze_price_action(df, cfg["india"])
        bias = analysis["bias"]
        if bias == "NEUTRAL":
            continue
        # enforce market bias
        market_bias = analyze_price_action(index_df, cfg["india"])['bias']
        if market_bias != 'NEUTRAL' and bias != market_bias:
            continue
        # additional filters
        if bias == 'LONG' and rs < cfg["india"].get("min_rs_threshold", 0.0):
            continue
        if bias == 'SHORT' and rs > -cfg["india"].get("min_rs_threshold", 0.0):
            continue

        candidates.append({
            "ticker": ticker.replace('.NS', ''),
            "bias": bias,
            "rs": rs,
            "vol_ratio": vol_ratio,
            "entry": analysis["entry"],
            "stop": analysis["stop"]
        })
    except Exception:
        log.exception("Error processing %s", ticker)
        continue

if not candidates:
    log.info("No candidates found")
    return pd.DataFrame(), [], "No candidates"

dfc = pd.DataFrame(candidates)
# scoring with weights
rs_w = cfg["india"].get("rs_weight", 1.0)
vol_w = cfg["india"].get("vol_weight", 1.0)
dfc["score"] = (dfc["rs"].abs() * 100 * rs_w) + (dfc["vol_ratio"] * 10 * vol_w)
dfc = dfc.sort_values("score", ascending=False)

picks = dfc.head(5).to_dict("records")
log.info("Found %d picks", len(picks))
return dfc, picks, f"Scan @ {scan_time.strftime('%H:%M')}"

---------- CLI / Main ----------

def main(): parser = argparse.ArgumentParser() parser.add_argument("--hour", type=int) parser.add_argument("--minute", type=int) parser.add_argument("--test", action="store_true") args = parser.parse_args()

cfg = load_cfg()
capital = cfg.get("capital_inr", 100000)

# Run scan
dfc, picks, window_desc = pick_signals(cfg, hour_of_day=args.hour, minute_of_hour=args.minute)

if not picks:
    msg = f"No clean setups found - {window_desc}"
    log.info(msg)
    if not args.test:
        send_tg(msg)
    return

lines = [f"🎯 Top picks - {datetime.now(IST).strftime('%d-%b-%Y %H:%M')} - {window_desc}"]

for p in picks:
    lvl = build_levels(p, cfg, capital)
    if lvl.get("qty", 0) <= 0:
        lines.append(f"{p['ticker']} {p['bias']} - Entry {lvl.get('entry')} Stop {lvl.get('stop')} - Qty insufficient for lot size")
        continue
    rr = abs((lvl['target'] - lvl['entry']) / max(abs(lvl['entry'] - lvl['stop']), 1e-6))
    lines.append(f"{p['ticker']} <b>{p['bias']}</b> Entry: {lvl['entry']:.2f} | Stop: {lvl['stop']:.2f} | Target: {lvl['target']:.2f} | Qty: {lvl['qty']} | R:R≈{rr:.2f} | RS:{p['rs']:.3f} Vol:{p['vol_ratio']:.2f}x")

result_msg = "\n".join(lines)
log.info("Sending results")
if not args.test:
    send_tg(result_msg)
else:
    # print to console for test mode
    print(result_msg)

if name == 'main': try: main() except Exception as e: log.exception("Fatal error in scanner: %s", e) try: send_tg(f"⚠️ Scanner Error: {e}") except Exception: pass sys.exit(1)

