# Stock Scanner Specification

**Project:** MRS Dashboard Stock Scanner  
**Date:** July 19, 2026  
**Status:** Planning / Brainstorm

---

## Overview

Build a stock scanner integrated with the MRS Dashboard that identifies RS leaders with strong fundamentals and favorable technical setups.

---

## Scan Criteria

### Technical Filters

| Filter | Condition | Notes |
|--------|-----------|-------|
| Uptrend structure | 50 SMA > 200 SMA | Golden cross / structural uptrend |
| Near trend | Price > 50 SMA | Not lagging |
| Not overextended | Within 20% of ATH | Room to run, not chasing |
| Relative Strength | RS vs SPY > 0 | Outperforming market |
| Volume confirmation | Volume vs 20d avg | Institutional interest |

### Fundamental Filters (CAN SLIM)

| Filter | Condition | Source |
|--------|-----------|--------|
| C - Current EPS | Quarterly EPS YoY growth 25%+ | Yahoo Finance |
| A - Annual EPS | Annual EPS growth 25%+ | Yahoo Finance |
| Revenue Growth | Quarterly revenue YoY growth 25%+ | Yahoo Finance |

### Chart Patterns

| Pattern | Detection Method | Difficulty |
|---------|------------------|------------|
| Tight consolidation | ATR compression over 10-20 days | Easy |
| Breakout from range | Price > X-day high | Easy |
| Higher lows | 3+ swing points trending up | Medium |

---

## Open Questions (To Decide)

1. **Universe:** 
   - [ ] All S&P 500 (~500 stocks)
   - [ ] Top 3 RS sectors only (~150 stocks, faster)
   - [ ] Custom watchlist

2. **Thresholds:**
   - [ ] Strict CAN SLIM: 25%+ EPS/Revenue growth
   - [ ] Flexible: 15%+ EPS/Revenue growth

3. **Output Format:**
   - [ ] Single ranked list with composite score
   - [ ] Separate lists by pattern type ("Tight bases", "Breakouts", "RS leaders")

4. **Weighting:**
   - [ ] Equal weight technicals and fundamentals
   - [ ] Prioritize one over the other

---

## Technical Implementation

### Data Sources

- **Price/Volume:** Yahoo Finance (yfinance)
- **Fundamentals:** Yahoo Finance quarterly financials
- **Sector membership:** Pre-defined mapping or Yahoo Finance sector info

### Constraints

- Yahoo Finance free tier — batch requests, ~50-100 stocks at a time
- S&P 500 full scan: ~2-3 minutes
- Run nightly after market close, cache results

### Architecture

```
StockScanner/
├── scanner.py          # Core scanning logic
├── filters.py          # Technical and fundamental filters
├── patterns.py         # Chart pattern detection
├── universe.py         # Stock universe management (S&P 500 list)
├── cache/              # Cached scan results
│   └── scan_results.csv
└── SCANNER_SPEC.md     # This file
```

### Dashboard Integration

- New section in app.py or separate page
- Display scan results as sortable/filterable table
- Link to individual stock charts (TradingView?)

---

## Reference: CAN SLIM Criteria

From William O'Neil's methodology:

- **C** - Current quarterly earnings per share: up 25%+ YoY
- **A** - Annual earnings growth: 25%+ over last 3-5 years
- **N** - New product, management, or price high
- **S** - Supply and demand: volume on breakouts
- **L** - Leader or laggard: RS rank 80+
- **I** - Institutional sponsorship: funds owning
- **M** - Market direction: follow the general market (MRS!)

Our scanner covers: C, A, S, L, and M (via MRS integration)

---

## Next Steps

1. Decide on universe and thresholds
2. Build core scanner.py with technical filters
3. Add fundamental filters (EPS/Revenue)
4. Add pattern detection
5. Integrate with dashboard
6. Test and refine

---

## Session Notes

**Brainstorm Date:** July 19, 2026

Key decisions made:
- Use SPY for volume (consistent with MRS calculations)
- End-of-day scan is sufficient (no real-time needed)
- Focus on patterns that don't require shape recognition
- ATR compression for tight bases
- Swing point detection for higher lows
