"""Test script to verify MRS scoring with today's inputs."""
import pandas as pd
import numpy as np
from datetime import date
import pipeline
import yfinance as yf

# Load existing history
hist = pipeline.load_history('mrs_history.csv')
print(f"CSV last date: {hist['date'].max().date()}")
print(f"Today: {date.today()}")
print(f"Rows in history: {len(hist)}")

# Your inputs for today
b20 = 56.5
adl = 1845.27  # TradingView scale
zg = 7496.59
pc = 0.76

# Fetch market data
today = pd.Timestamp(date.today())
print("\nFetching market data...")

market_data = {}
for field, ticker in [('spy', 'SPY'), ('spx', '^GSPC'), ('vix', '^VIX'), ('skew', '^SKEW')]:
    try:
        h = yf.Ticker(ticker).history(period='5d', auto_adjust=True)
        if not h.empty:
            h.index = pd.to_datetime(h.index).normalize().tz_localize(None)
            latest = h['Close'].iloc[-1]
            latest_date = h.index[-1].date()
            market_data[field] = latest
            print(f"  {field.upper()}: {latest:.2f} (as of {latest_date})")
    except Exception as e:
        print(f"  {field.upper()}: ERROR - {e}")

# Get last row's Phi values
last = hist.iloc[-1]
last_date = pd.Timestamp(last['date']).date()
print(f"\nLast CSV row: {last_date}")
print(f"  VIX Phi:  {last['vix_phi']:.3f}")
print(f"  Ext Phi:  {last['ext_phi']:.3f}")
print(f"  Mom Phi:  {last['mom_phi']:.3f}")
print(f"  B20 Phi:  {last['b20_phi']:.3f}")
print(f"  ADL Phi:  {last['adl_phi']:.3f}")
print(f"  SKEW Phi: {last['skew_phi']:.3f}")

# Score each component using last row's Phi (they're stable day-to-day)
vs, vst = pipeline.score_vix(last['vix_phi'])
ms, mst = pipeline.score_momentum(last['mom_phi'])
es, est = pipeline.score_extension(last['ext_phi'], last['mom_phi'])
as_, ast = pipeline.score_adl(last['adl_phi'])
bs, bst = pipeline.score_b20(last['b20_phi'], last['adl_phi'])

# PC scoring with your new value
pc_sma10 = hist['pc_ratio'].rolling(10).mean().iloc[-1]
ps, pst = pipeline.score_pc(pc, pc_sma10)

# SKEW scoring
ss, sst = pipeline.score_skew(last['skew_phi'], pc)

# Gamma with your new ZG and today's SPX
spx_val = market_data.get('spx', last.get('spx', np.nan))
gs, gst = pipeline.score_gamma(spx_val, zg)

print(f"\n--- Component Scores (with your inputs) ---")
print(f"  VIX:       {vs:+.1f} ({vst})")
print(f"  Extension: {es:+.1f} ({est})")
print(f"  Momentum:  {ms:+.1f} ({mst})")
print(f"  ADL:       {as_:+.1f} ({ast})")
print(f"  B20:       {bs:+.1f} ({bst})")
print(f"  PC Ratio:  {ps:+.1f} ({pst})")
print(f"  SKEW:      {ss:+.1f} ({sst})")
print(f"  Gamma:     {gs:+.1f} ({gst}) [SPX={spx_val:,.0f} vs ZG={zg:,.0f}]")

total = vs + es + ms + as_ + bs + ps + ss + gs
regime = pipeline.regime_label(total)
print(f"\n{'='*50}")
print(f"  MRS SCORE: {total:+.2f}  -->  {regime}")
print(f"{'='*50}")

# Show what changed from yesterday
old_mrs = last['mrs_score']
print(f"\nYesterday's MRS: {old_mrs:+.2f}")
print(f"Change: {total - old_mrs:+.2f}")
