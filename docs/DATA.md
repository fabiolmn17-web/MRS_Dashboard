# Data

## Files in the repository

| File | Content | Written by | When | Read by |
|---|---|---|---|---|
| `mrs_history.csv` | One row per NYSE session since 2005: SPY, SPX, VIX, SKEW, put/call, ADL, B20, zero gamma, volume; derived signals, Phi values, component scores, MRS score; forward-log columns (pedal, cell, effective bets, insurance cost, allocation tickets) | `update_data.py`, `backfill.py` | Daily after the close; manual inputs | `app.py` (raw GitHub URL; fallback: bundled copy) |
| `qqq_mrs_history.csv` | QQQ version of the history (research) | maintenance scripts | Occasional | Research |
| `StockScanner/output/scan_results.csv` | One row per scanned stock: technicals, RS, momentum fields, Deep dip / pullback setups, bases, breakouts, heat score, market label (`mkt_state`, `qqq_state`), `data_through` | `run_scanner.py` | 22:00 UTC and 08:30 UTC | Scanner section, leaders, context |
| `StockScanner/output/scan_coverage_low.txt` | Present only when the last scan was not published (coverage below 70% of the previous one); says why | `scanner.py` | With the scan | Load-health line |
| `StockScanner/output/industry_rs.csv`, `industry_holdings.csv` | Industry ETFs: RS, labels, `last_bar`; top-10 holdings | `run_industry.py` | With the scan | Leaders / avoid, sector table, industry cells |
| `gamma_shadow.csv` | SPY 40-day zero gamma, net GEX from Gamma Live | `gamma_shadow.py` | After the close (trading PC) | Decision strip ("Zero γ GL (shadow)") |
| `watchlist` branch: `watchlist.csv` | Tickers, trigger, stop, note | the app (GitHub API, token in secrets) | On save | Watchlist |

## External sources

| Source | Used for | Failure mode | Guard |
|---|---|---|---|
| Yahoo Finance (yfinance) | SPY, SPX, VIX, SKEW, volume; stock and ETF prices; live series in the app | Throttling, late daily bars (the session's bars often missing before about 02:00 UTC), stale subsets in batches | Retries, single-ticker re-fetch, morning scan, coverage guard, `data_through` |
| CBOE | Put/call ratio | Page change, delay | Manual input fallback |
| InsiderFinance (public page) | SPX zero gamma | Pattern not found | Manual input; Gamma Live shadow |
| TradingView (manual) | ADL, B20 (S5TW) when typed in | Not entered | Carried forward; retroactive fill |
| GitHub raw URL | The app's copy of the history | Fetch failure | Bundled copy, flagged in the health line |

## Conventions

- **Dates:** New York trading sessions. A run started after midnight UTC belongs to the previous New York session (`auto_fetch._session_date`).
- **Prices:** dividend-adjusted for stocks in the scanner (yfinance `auto_adjust=True`). Split-adjusted, not dividend-adjusted, for the allocation tickets (as on TradingView).
- **ATR:** Wilder ATR(14) everywhere for stocks and the pedal. ATR(10) only in the allocation tickets' stop, as tested.
- **Percentiles (Phi):** 3-year (756-session) rolling, past-only.
