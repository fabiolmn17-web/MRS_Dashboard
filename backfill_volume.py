"""Backfill volume data into mrs_history.csv"""
import pandas as pd
import yfinance as yf
import numpy as np

# Load history
hist = pd.read_csv('mrs_history.csv', parse_dates=['date'])
print(f"History rows: {len(hist)}")
print(f"Date range: {hist['date'].min()} to {hist['date'].max()}")

# Get SPY volume for full history
start = hist['date'].min().strftime('%Y-%m-%d')
spy = yf.Ticker('SPY').history(start=start, auto_adjust=True)
spy.index = pd.to_datetime(spy.index).normalize().tz_localize(None)
print(f"SPY volume rows: {len(spy)}")

# Create volume series with date index
vol_series = spy['Volume'].copy()
vol_series.index = vol_series.index.normalize()

# Merge volume into hist
hist = hist.set_index('date')
hist.index = pd.to_datetime(hist.index).normalize()
hist['volume'] = vol_series
hist = hist.reset_index()

# Compute volume divergence signals
hist['price_60d_chg'] = hist['spy'].pct_change(60)
hist['vol_60d_chg'] = hist['volume'].pct_change(60)
hist['vol_divergence'] = ((hist['price_60d_chg'] > 0) & (hist['vol_60d_chg'] < -0.10)).astype(int)

# Check results
vol_filled = hist['volume'].notna().sum()
print(f"Volume filled: {vol_filled} rows ({vol_filled/len(hist)*100:.1f}%)")
print(f"Volume divergence signals: {hist['vol_divergence'].sum()}")
print(f"Last 5 rows:")
print(hist[['date', 'spy', 'volume', 'price_60d_chg', 'vol_60d_chg', 'vol_divergence']].tail())

# Save
hist.to_csv('mrs_history.csv', index=False)
print("\nSaved mrs_history.csv!")
