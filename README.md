# India Intraday Market Scanner

Automated NSE stock scanner that runs every hour during market hours to identify high-probability trading opportunities.

## Features

- Runs automatically on market days from 9:10 AM to 3:10 PM IST
- Analyzes the opening range (9:15-9:45) for breakout setups
- Hourly updates throughout the trading day
- Scans the most active 100 NSE stocks
- Identifies setups aligned with the NIFTY index direction
- Calculates optimal position sizing based on risk parameters
- Delivers results via Telegram notifications

## How It Works

1. **Morning Scan (9:10 AM)**: Analyzes pre-market activity and opening range
2. **Hourly Scans (10:10, 11:10, ...)**: Provides hourly updates on new setups
3. **Relative Strength**: Finds stocks outperforming/underperforming the index
4. **Volume Analysis**: Identifies stocks with above-average trading volume
5. **Risk Management**: Calculates precise entry, stop, and target levels
6. **Position Sizing**: Determines optimal quantity based on your capital

## Setup

1. Fork this repository
2. Add your Telegram credentials as repository secrets:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Adjust configuration in `config.json` if needed
4. GitHub Actions will automatically run the scanner on schedule

## Configuration

Edit `config.json` to customize scanner behavior:

```json
{
    "india": {
        "dynamic_universe_size": 100,
        "risk_pct": 0.5,
        "buffer_bp": 5,
        "index": "^NSEI"
    },
    "capital_inr": 100000
}
