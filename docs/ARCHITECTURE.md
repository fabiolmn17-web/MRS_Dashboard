# Architecture

## Overview

```
          ┌────────────────────── GitHub repository MRS_Dashboard (main) ──────────────────────┐
          │                                                                                      │
 Yahoo ──►│ update_data.py (daily)  ──► mrs_history.csv ──┐                                      │
 CBOE  ──►│   pipeline.py (scoring), auto_fetch.py        │                                      │
 Inside-  │                                               │                                      │
 Finance ►│ backfill.py (manual form / dashboard inputs) ─┤                                      │
          │                                               ▼                                      │
 Yahoo ──►│ StockScanner/run_scanner.py ──► output/scan_results.csv ──┐                          │
          │ StockScanner/run_industry.py ─► output/industry_*.csv ────┤                          │
          │                                                           ▼                          │
 Gamma    │ gamma_shadow.py (triggered from the trading PC) ──► gamma_shadow.csv                 │
 Live PC ►│                                                           │                          │
          └───────────────────────────────────────────────────────────┼──────────────────────────┘
                                                                      ▼
                         Streamlit Cloud: app.py (+ pages/2_Journal.py)  ◄── Yahoo (live: sectors, SPY/QQQ, VIX3M)
                                                                      │
                                                                      ▼
                                                               the dashboard
```

The dashboard reads the committed files (the MRS history from GitHub's raw URL, falling back to the bundled copy), plus a few live Yahoo series cached 1–3 hours. It writes only the watchlist (to the `watchlist` branch, with a token in the Streamlit secrets).

## Components

| Module | Role |
|---|---|
| `app.py` | The page: decision strip, load-health line, Market context card, Risk today panel, allocation card, leaders / avoid, rotation, sector table, scanner, watchlist, input form, tabs. |
| `pipeline.py` | MRS scoring: derived signals, 3-year rolling percentiles (Phi), component states and weights, regime bands. `SCORING_VERSION`. History update (`update_history`). |
| `auto_fetch.py` | Automatic inputs: zero gamma (InsiderFinance page), ADL, B20, put/call; session date (New York). |
| `update_data.py` | The daily batch (GitHub Actions): NYSE trading-day check, fetch, append, rescore, forward logs (risk panel, allocation), save. |
| `backfill.py` | Manual inputs from the dashboard form (via the backfill workflow); fills missing sessions. |
| `market_state.py` | Decision strip HTML, band actions, leaders / avoid lists. |
| `risk_panel.py` | Risk today: pedal (EASE OFF / NORMAL / DIP ZONE), today's state, effective bets, insurance cost, volatility guide. |
| `allocation.py` | SPY / QQQ allocation tickets (rebound A, breakout B): replay, state, card, forward log. |
| `context_card.py` | Market context card: SPY phase (dip zone → recovery → trend → out), the regime × phase cell, where to look, trade management. |
| `scan_view.py`, `scan_table/` | Scanner table cells (setup, breakout, grey-out, chasing tag) and the click-to-chart component. |
| `chart_view.py`, `watchlist.py`, `wl_table/` | Charts and the watchlist (stored on the `watchlist` branch). |
| `rs_labels.py`, `sector_view.py`, `rotation_view.py`, `heat_score.py` | Relative-strength labels, sector table, rotation map, heat score. |
| `gamma_shadow.py` | Stores the Gamma Live 40-day zero gamma (shadow only; not in the score). |
| `StockScanner/` | Nightly scanner: universe, prices, technicals, RS, momentum fields, Deep dip, pullbacks, bases (`bases.py`: patterns, follow-through day, market label), fundamentals, industries. |
| `pages/2_Journal.py` | Trading journal page. |
| `atn_brand.py` | Brand: fonts, CSS, plotly template. |
| `backtest_mrs.py`, `backtest_qqq.py`, `recompute_history.py`, `apply_v*_rescore.py` | Research and one-off maintenance scripts; not used by the live app. |

## Schedules (GitHub Actions, times UTC)

| Workflow | When | Writes |
|---|---|---|
| MRS Daily Update (`daily_update.yml`) | 21:30 Mon–Fri (often starts later) | `mrs_history.csv` |
| CAN SLIM Scanner (`scanner.yml`) | 22:00 Mon–Fri and 08:30 Tue–Sat (09:30 Berlin in winter; change to 07:30 at the end of March) | `StockScanner/output/scan_results.csv`, `scan_coverage_low.txt`, `industry_rs.csv`, `industry_holdings.csv` |
| MRS Backfill (`backfill.yml`) | Manual (dashboard input form) | `mrs_history.csv` |
| Gamma Live zero gamma (`gamma_shadow.yml`) | Manual, triggered by the trading PC after the close | `gamma_shadow.csv` |
| EP scanner (`ep_scanner.yml`) | Schedule disabled | `StockScanner/output/ep_results.csv` |

## Reliability guards

| Guard | Where |
|---|---|
| NYSE trading-day calendar (Columbus / Veterans Day open; Good Friday by rule) | `update_data.py` |
| Pull before push | `daily_update.yml`, `scanner.yml` |
| Gamma shadow concurrency group | `gamma_shadow.yml` |
| Scanner retry of failed downloads; coverage guard (keep the previous scan below 70% coverage) | `StockScanner/scanner.py` |
| Stale-bar re-download and freshness check (`data_through`) | `StockScanner/scanner.py` |
| Load-health line (failed or stale sections listed under the decision strip) | `app.py` |
| GitHub fetch fallback to the bundled history (flagged in the health line) | `app.py` |
