#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
India Intraday Scanner - GitHub Actions Version
"""

import os
import sys
import json
import pytz
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path

# Configure logging
LOG_DIR = "logs"
Path(LOG_DIR).mkdir(exist_ok=True)
log_file = Path(LOG_DIR) / f"scanner_{dt.datetime.now().strftime('%Y%m%d')}.log"

def log(message, level="INFO"):
    """Log message to file and console"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {level}: {message}"
    
    # Print to console (will appear in GitHub Actions logs)
    print(log_message)
    
    # Write to log file
    with open(log_file, "a") as f:
        f.write(log_message + "
")

# Constants
IST = pytz.timezone('Asia/Kolkata')
CONFIG_FILE = "config.json"

def load_cfg():
    """Load configuration from JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        log(f"Configuration loaded from {CONFIG_FILE}")
        return cfg
    except Exception as e:
        log(f"Failed to load configuration: {e}", "ERROR")
        # Default configuration if file not found
        return {
            "india": {
                "dynamic_universe_size": 100,
                "risk_pct": 0.5,
                "buffer_bp": 5,
                "min_rs_threshold": 0.85,
                "min_volume_impulse": 1.5,
                "index": "^NSEI"
            },
            "capital_inr": 100000
        }

def is_trading_day(date):
    """Check if given date is a trading day (not weekend or holiday)"""
    # NSE holidays for 2023-2025 (incomplete list - update as needed)
    holidays = [
        # 2024 holidays
        dt.date(2024, 1, 26),  # Republic Day
        dt.date(2024, 3, 25),  # Holi
        dt.date(2024, 3, 29),  # Good Friday
        dt.date(2024, 4, 17),  # Ram Navami
        dt.date(2024, 5, 1),   # Maharashtra Day
        dt.date(2024, 8, 15),  # Independence Day
        dt.date(2024, 10, 2),  # Gandhi Jayanti
        dt.date(2024, 12, 25), # Christmas
    ]
    
    # Check if it's a weekend
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        log(f"{date} is a weekend")
        return False
    
    # Check if it's a holiday
    if date in holidays:
        log(f"{date} is a market holiday")
        return False
    
    return True

def is_market_open(current_time=None):
    """Check if market is currently open"""
    if current_time is None:
        current_time = dt.datetime.now(IST)
    
    # Market hours: 9:15 AM - 3:30 PM
    market_open = dt.time(9, 15)
    market_close = dt.time(15, 30)
    
    # Check if current time is within market hours
    if market_open <= current_time.time() <= market_close:
        # Also check if it's a trading day
        if is_trading_day(current_time.date()):
            return True
    
    return False

def fetch_active_stocks(limit=100, index="^NSEI"):
    """Fetch the most active NSE stocks"""
    log(f"Fetching {limit} most active NSE stocks")
    
    try:
        # Method 1: Get components from Nifty 50/100/200
        if index == "^NSEI":  # NIFTY 50
            nifty_components = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "ITC", "SBIN", "BAJFINANCE", "BHARTIARTL",
                "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
                "TITAN", "SUNPHARMA", "TATAMOTORS", "WIPRO", "ULTRACEMCO",
                "ADANIENT", "JSWSTEEL", "TATASTEEL", "HCLTECH", "NTPC",
                "BAJAJFINSV", "M&M", "NESTLEIND", "ADANIPORTS", "POWERGRID",
                "ONGC", "GRASIM", "HDFCLIFE", "DIVISLAB", "INDUSINDBK",
                "SBILIFE", "TECHM", "DRREDDY", "EICHERMOT", "COALINDIA",
                "BAJAJ-AUTO", "HINDALCO", "TATACONSUM", "APOLLOHOSP", "BRITANNIA",
                "UPL", "CIPLA", "BPCL", "HEROMOTOCO", "LTIM"
            ]
        else:  # Fallback to top liquid stocks
            nifty_components = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "ITC", "SBIN", "BAJFINANCE", "BHARTIARTL",
                "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI"
            ]
        
        # Add .NS suffix for Yahoo Finance
        return [f"{s}.NS" for s in nifty_components[:limit]]
        
    except Exception as e:
        log(f"Error fetching active stocks: {e}", "ERROR")
        # Return a small subset of major stocks as fallback
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

def yf_intraday(tickers, period="1d", interval="1m"):
    """Fetch intraday data for the given tickers"""
    try:
        log(f"Fetching data for {len(tickers)} tickers: period={period}, interval={interval}")
        
        # Split into batches to avoid request timeouts (10 tickers per batch)
        all_data = {}
        batch_size = 10
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=False,
                    progress=False
                )
                
                # Process batch results
                if len(batch) == 1:
                    # Handle single ticker case
                    ticker = batch[0]
                    if not data.empty:
                        all_data[ticker] = data
                else:
                    # Handle multi-ticker case
                    for ticker in batch:
                        if ticker in data.columns.levels[0]:
                            ticker_data = data[ticker].copy()
                            if not ticker_data.empty:
                                all_data[ticker] = ticker_data
                
            except Exception as e:
                log(f"Error fetching batch {i}-{i+batch_size}: {e}", "ERROR")
        
        return all_data
    
    except Exception as e:
        log(f"Error in yf_intraday: {e}", "ERROR")
        return {}

def vwap(df):
    """Calculate VWAP (Volume Weighted Average Price)"""
    try:
        if df.empty or 'Volume' not in df.columns:
            return pd.Series(index=df.index)
            
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    except Exception as e:
        log(f"Error calculating VWAP: {e}", "ERROR")
        return pd.Series(index=df.index)

def opening_range(df, market_open_time=dt.time(9, 15), window_minutes=30):
    """Calculate opening range high and low"""
    try:
        if df.empty:
            return np.nan, np.nan
            
        # Convert index to IST if it has timezone info
        if df.index.tzinfo is not None:
            df_ist = df.tz_convert(IST)
        else:
            # If no timezone info, assume UTC and convert to IST
            df_ist = df.tz_localize('UTC').tz_convert(IST)
        
        # Calculate market open and window end time
        today = df_ist.index[-1].date()
        market_open = dt.datetime.combine(today, market_open_time).replace(tzinfo=IST)
        window_end = market_open + dt.timedelta(minutes=window_minutes)
        
        # Filter data to opening range window
        window_data = df_ist[(df_ist.index >= market_open) & (df_ist.index < window_end)]
        
        if window_data.empty:
            return np.nan, np.nan
            
        return float(window_data['High'].max()), float(window_data['Low'].min())
    
    except Exception as e:
        log(f"Error calculating opening range: {e}", "ERROR")
        return np.nan, np.nan

def rs_vs_index(stock_df, index_df):
    """Calculate relative strength vs index"""
    try:
        if stock_df.empty or index_df.empty:
            return 0.0
            
        # Try to get first and last prices
        stock_first = stock_df['Close'].iloc[0]
        stock_last = stock_df['Close'].iloc[-1]
        index_first = index_df['Close'].iloc[0]
        index_last = index_df['Close'].iloc[-1]
        
        # Calculate returns
        stock_return = stock_last / stock_first - 1
        index_return = index_last / index_first - 1
        
        # Calculate relative strength
        rs = stock_return - index_return
        
        return rs
    
    except Exception as e:
        log(f"Error calculating RS: {e}", "ERROR")
        return 0.0

def analyze_volume(df):
    """Calculate volume impulse (ratio of current volume to average)"""
    try:
        if df.empty or 'Volume' not in df.columns:
            return 1.0
            
        # Calculate average volume (excluding most recent bars)
        recent_volume = df['Volume'].iloc[-4:].mean()  # Last 4 bars
        overall_volume = df['Volume'].mean()
        
        # Avoid division by zero
        if overall_volume == 0:
            return 1.0
            
        return float(recent_volume / overall_volume)
    
    except Exception as e:
        log(f"Error analyzing volume: {e}", "ERROR")
        return 1.0

def build_levels(p, risk_pct, capital, buffer_bp=5):
    """Calculate entry, stop, and target levels with position sizing"""
    try:
        if p['bias'] == 'LONG':
            entry = p['entry'] * (1 + buffer_bp/10000.0)
            stop = p['stop'] * (1 - buffer_bp/10000.0)
            risk = max(entry - stop, 0.01)
            qty = max(0, int((capital * risk_pct/100) / risk))
            tgt = entry + 2 * risk  # 2:1 reward-to-risk ratio
        elif p['bias'] == 'SHORT':
            entry = p['entry'] * (1 - buffer_bp/10000.0)
            stop = p['stop'] * (1 + buffer_bp/10000.0)
            risk = max(stop - entry, 0.01)
            qty = max(0, int((capital * risk_pct/100) / risk))
            tgt = entry - 2 * risk  # 2:1 reward-to-risk ratio
        else:
            entry = stop = tgt = 0
            qty = 0
            
        return entry, stop, tgt, qty
    
    except Exception as e:
        log(f"Error building levels: {e}", "ERROR")
        return 0, 0, 0, 0

def send_tg(msg):
    """Send message to Telegram"""
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        log("Telegram credentials not set", "WARNING")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    try:
        payload = {
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            log("Telegram message sent successfully")
        else:
            log(f"Failed to send Telegram message: {response.text}", "ERROR")
    
    except Exception as e:
        log(f"Error sending Telegram message: {e}", "ERROR")

def analyze_price_action(data, timeframe="hourly"):
    """Analyze price action to determine bias"""
    try:
        if data.empty:
            return {"bias": "NEUTRAL", "entry": 0, "stop": 0}
            
        # Get closing prices
        closes = data['Close']
        if len(closes) < 2:
            return {"bias": "NEUTRAL", "entry": 0, "stop": 0}
            
        latest_close = closes.iloc[-1]
        prev_close = closes.iloc[-2]
        highest_high = data['High'].max()
        lowest_low = data['Low'].min()
        
        # Get VWAP
        vwap_series = vwap(data)
        latest_vwap = vwap_series.iloc[-1] if not vwap_series.empty else latest_close
        
        # Calculate opening range
        orh, orl = opening_range(data)
        
        # Determine bias
        if np.isnan(orh) or np.isnan(orl):
            # Fallback if opening range is not available
            if latest_close > prev_close and latest_close > latest_vwap:
                bias = "LONG"
                entry = latest_close
                stop = lowest_low
            elif latest_close < prev_close and latest_close < latest_vwap:
                bias = "SHORT"
                entry = latest_close
                stop = highest_high
            else:
                bias = "NEUTRAL"
                entry = stop = latest_close
        else:
            # Use opening range
            if latest_close > orh and latest_close > latest_vwap:
                bias = "LONG"
                entry = orh
                stop = orl
            elif latest_close < orl and latest_close < latest_vwap:
                bias = "SHORT"
                entry = orl
                stop = orh
            else:
                bias = "NEUTRAL"
                entry = stop = latest_close
                
        return {
            "bias": bias,
            "entry": float(entry),
            "stop": float(stop)
        }
    
    except Exception as e:
        log(f"Error analyzing price action: {e}", "ERROR")
        return {"bias": "NEUTRAL", "entry": 0, "stop": 0}

def pick_signals(cfg, hour_of_day=None, minute_of_hour=None):
    """Scan for trading signals based on time of day"""
    # Get current date in IST
    ist_now = dt.datetime.now(IST)
    today = ist_now.date()
    
    # Use provided hour/minute or current time
    if hour_of_day is None:
        hour_of_day = ist_now.hour
    if minute_of_hour is None:
        minute_of_hour = ist_now.minute
        
    # Create a datetime for the specified hour/minute
    scan_time = dt.datetime.combine(today, dt.time(int(hour_of_day), int(minute_of_hour))).replace(tzinfo=IST)
    
    log(f"Running scan for time: {scan_time.strftime('%Y-%m-%d %H:%M')} IST")
    
    # Check if it's a trading day
    if not is_trading_day(today):
        log(f"Today {today} is not a trading day")
        return pd.DataFrame(), [], "Non-trading day"
    
    # Configure time window based on hour of day
    if int(hour_of_day) == 9:
        # Morning scan (9:10 AM) - focus on opening range
        window_desc = "Opening Range (9:15-9:45)"
        period = "1d"
        interval = "1m"
    else:
        # Hourly scans - analyze more recent data
        window_desc = f"Hourly Scan ({int(hour_of_day)-1}:00-{hour_of_day}:00)"
        period = "1d"
        interval = "5m"
    
    # Get active stocks for today
    dynamic_limit = cfg["india"]["dynamic_universe_size"]
    active_stocks = fetch_active_stocks(limit=dynamic_limit)
    
    # Get index data first (NIFTY 50)
    index = cfg["india"].get("index", "^NSEI")
    index_data = yf.download(index, period=period, interval=interval, progress=False)
    
    if index_data.empty:
        log("Failed to fetch index data", "ERROR")
        return pd.DataFrame(), [], "Index data unavailable"
    
    # Analyze index to get market bias
    index_analysis = analyze_price_action(index_data)
    market_bias = index_analysis["bias"]
    log(f"Market bias from index {index}: {market_bias}")
    
    # Fetch data for all stocks
    stock_data = yf_intraday(active_stocks, period=period, interval=interval)
    
    candidates = []
    
    # Process each stock
    for ticker, data in stock_data.items():
        try:
            if data.empty or len(data) < 5:
                continue
                
            # Calculate relative strength
            rs = rs_vs_index(data, index_data)
            
            # Analyze volume
            vol_ratio = analyze_volume(data)
            
            # Analyze price action
            analysis = analyze_price_action(data)
            bias = analysis["bias"]
            
            # Skip neutral setups
            if bias == "NEUTRAL":
                continue
                
            # Only consider setups aligned with market bias
            if market_bias != "NEUTRAL" and bias != market_bias:
                continue
                
            # Additional filters
            if bias == "LONG" and rs < 0:
                continue
            if bias == "SHORT" and rs > 0:
                continue
                
            # Add to candidates list
            ticker_name = ticker.replace(".NS", "")
            candidates.append({
                "ticker": ticker_name,
                "bias": bias,
                "rs": rs,
                "vol_ratio": vol_ratio,
                "entry": analysis["entry"],
                "stop": analysis["stop"]
            })
            
        except Exception as e:
            log(f"Error processing {ticker}: {e}", "ERROR")
    
    # Create DataFrame
    dfc = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    
    if dfc.empty:
        log("No candidates found")
        return dfc, [], window_desc
    
    # Score and rank candidates
    dfc["score"] = dfc["rs"].abs() * 100 + dfc["vol_ratio"] * 10
    dfc = dfc.sort_values("score", ascending=False)
    
    # Select top picks
    picks = dfc.head(5).to_dict("records")
    log(f"Found {len(picks)} qualified setups")
    
    return dfc, picks, window_desc

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="India Intraday Scanner")
    parser.add_argument("--hour", type=int, help="Hour of day (24-hour format)")
    parser.add_argument("--minute", type=int, help="Minute of hour")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_cfg()
    
    # Get current time in IST
    ist_now = dt.datetime.now(IST)
    hour = args.hour if args.hour is not None else ist_now.hour
    minute = args.minute if args.minute is not None else ist_now.minute
    
    # Check if it's a trading day
    if not is_trading_day(ist_now.date()) and not args.test:
        msg = "NSE Holiday/Closed: Today is not a trading day."
        log(msg)
        send_tg(msg)
        return
    
    # Check if market should be open (skip for test mode)
    if not args.test:
        scan_time = dt.datetime.combine(ist_now.date(), dt.time(hour, minute)).replace(tzinfo=IST)
        if hour < 9 or (hour == 9 and minute < 15) or hour > 15 or (hour == 15 and minute > 30):
            msg = f"Market closed at {scan_time.strftime('%H:%M')} IST. Regular hours: 09:15-15:30."
            log(msg)
            if hour >= 9:  # Only send message during daytime
                send_tg(msg)
            return
    
    # Let user know scan is starting
    scan_start_msg = f"🔍 Scanning top {cfg['india']['dynamic_universe_size']} NSE stocks at {hour}:{minute:02d} IST..."
    log(scan_start_msg)
    send_tg(scan_start_msg)
    
    # Run the scan
    table, picks, window_desc = pick_signals(cfg, hour, minute)
    
    if not picks:
        msg = f"No clean setups found in {window_desc}."
        log(msg)
        send_tg(msg)
        return
    
    # Generate results message
    risk_pct = cfg["india"].get("risk_pct", 0.5)
    capital = cfg.get("capital_inr", 100000)
    buf = cfg["india"].get("buffer_bp", 5)
    
    lines = [f"🎯 Top picks ({len(picks)}) - {ist_now.strftime('%d-%b-%Y %H:%M')} - {window_desc}:"]
    
    for p in picks:
        entry, stop, tgt, qty = build_levels(p, risk_pct, capital, buf)
        rr = abs((tgt-entry)/max(abs(entry-stop), 1e-6))
        
        lines.append(
            f"{p['ticker']} <b>{p['bias']}</b>
"
            f"  Entry: {entry:.2f} | Stop: {stop:.2f} | Target: {tgt:.2f} | Qty: {qty}
"
            f"  RS: {p['rs']:.2f} | Vol: {p['vol_ratio']:.1f}x | R:R≈{rr:.1f}x"
        )
    
    # Send results
    result_msg = "
".join(lines)
    log(f"Sending results with {len(picks)} picks")
    send_tg(result_msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        log(error_msg, "ERROR")
        send_tg(f"⚠️ Scanner Error: {error_msg}")
        sys.exit(1)
