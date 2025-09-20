#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekend Report Generator
"""

import os
import sys
import glob
import json
import pytz
import datetime as dt
import pandas as pd
import requests
from pathlib import Path

# Configure timezone
IST = pytz.timezone('Asia/Kolkata')

def send_tg(msg):
    """Send message to Telegram"""
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("Telegram credentials not set")
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
            print("Telegram message sent successfully")
        else:
            print(f"Failed to send Telegram message: {response.text}")
    
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def main():
    """Generate weekly report from logs"""
    # Get current date and week start/end
    now = dt.datetime.now(IST)
    week_end = now.date()
    week_start = (now - dt.timedelta(days=7)).date()
    
    print(f"Generating weekly report for {week_start} to {week_end}")
    
    # Find all log files
    log_files = glob.glob("logs/scanner_*.log")
    
    if not log_files:
        print("No log files found")
        msg = f"📊 Weekly Report ({week_start} to {week_end})
"
        msg += "No scanner activity detected this week.
"
        msg += "Please check GitHub Actions for any issues.
"
        msg += "Have a great weekend! 🌈"
        send_tg(msg)
        return
    
    # Parse logs to extract key information
    scan_stats = {
        "total_scans": 0,
        "successful_scans": 0,
        "error_count": 0,
        "signals": {
            "LONG": 0,
            "SHORT": 0
        }
    }
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if "Running scan for" in line:
                        scan_stats["total_scans"] += 1
                    if "Sending results with" in line:
                        scan_stats["successful_scans"] += 1
                    if "LONG" in line:
                        scan_stats["signals"]["LONG"] += 1
                    if "SHORT" in line:
                        scan_stats["signals"]["SHORT"] += 1
                    if "ERROR" in line:
                        scan_stats["error_count"] += 1
        except Exception as e:
            print(f"Error processing log file {log_file}: {e}")
    
    # Generate report
    msg = f"📊 <b>Weekly Scanner Report ({week_start} to {week_end})</b>
"
    msg += f"<b>Activity Summary:</b>
"
    msg += f"• Total scans run: {scan_stats['total_scans']}
"
    msg += f"• Successful scans: {scan_stats['successful_scans']}
"
    msg += f"• Error count: {scan_stats['error_count']}
"
    
    msg += f"<b>Signal Distribution:</b>
"
    msg += f"• LONG signals: {scan_stats['signals']['LONG']}
"
    msg += f"• SHORT signals: {scan_stats['signals']['SHORT']}
"
    
    msg += "The scanner is ready for next week's trading. Have a great weekend! 🌄"
    
    # Send report
    send_tg(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Critical error in weekend report: {str(e)}"
        print(error_msg)
        send_tg(f"⚠️ Weekend Report Error: {error_msg}")
        sys.exit(1)
