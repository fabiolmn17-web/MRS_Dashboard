# Changelog

Versions follow `docs/PROCESS.md`:
- MAJOR: scoring model or headline meaning;
- MINOR: new card, scan, rule or message;
- PATCH: fixes, wording, reliability.

The current version is in `VERSION` and in the page footer. Entries before 6.3.1 were reconstructed from the commit history and the project notes on 13 Nov 2026.

## 6.4.0 · 2026-11-14

Roadmap steps 1–3 (Claude Doc "MRS roadmap: from the Nov 2026 tests to the dashboard") and the page inventory (Step 0). The page now answers three questions in order: what can be done today, where to look, how to exit.

- **Today card** (new, `today_card.py`) replaces the decision strip, the Market context card, the pedal tile and the allocation card on the main path:
  - MRS score, band, DEEP badge, percentile, side;
  - SPY phase track (DIP ZONE → RECOVERY → TREND → OUT) and the levels that change it;
  - **This state:** EDGE / NO EDGE with the survivorship-free cell numbers (S3 panel B; about 0.05R lower than before);
  - **Risk per trade:** 1% on the risk-off side with SPY in DIP ZONE / RECOVERY / TREND, 0.7% risk-on with SPY in TREND, 0.5% elsewhere; open risk ≤ 3.5% of NLV counted to the current stops; ≤ 10 positions (P2, Provisional);
  - **Hard stop:** −15% from the account peak (Fabio's rule; P2 addendum on what it does and does not do);
  - **Where to look** (side-dependent: weaker names did better on the risk-off side) and **Exit** (2 ATR(14) ≤ 10%; hold the stop until +2R, then trail the 21 EMA);
  - alert lines only when active: EASE OFF, correlation stress, DEEP RISK-OFF, no shorts on laggards in RISK-OFF.
- **Folds:** Market detail (levels and changes, risk panel, SPY / QQQ tickets, leaders, avoid, rotation), Sectors and industries, Why the model says this. Sidebar: one "Show everything" switch replaces the Risk panel and Allocation toggles.
- **Context card:** cell numbers survivorship-free; "choose the name by leadership" replaced (it contradicted S3).
- **Scanner:**
  - long scans in order of the evidence: Deep dip, Pullbacks, Bases, Breakouts, Momentum list (All + Hot 50 merged, sorted by RS rating);
  - short scans folded into one group;
  - 12 default columns (Ticker, Setup, Base / pivot, Breakout, RS rtg, Ext, RVOL, Earnings, % 52W Hi, Squeeze, Industry, Heat); "More columns" shows all 21;
  - "More filters" fold (squeeze, base faults, correction, depth);
  - new chips: **✕ oversold** on a 50-day reclaim after RSI(14) < 35 in the 10 sessions before (R1, Provisional avoid); **breakout stop under the 5-day low** (S2, Tested);
  - context line: "Setups mark where the trade is wrong, not an edge".
- **Nightly scanner** (`StockScanner/momentum.py`): new fields `p50_os`, `p50_rsi10`, `low_5d`. The chips appear after the next scan.

## 6.3.2 · 2026-11-13

- **Load-health line** under the decision strip: "Data issues this run (n): section (reason)", or "✓ All sections loaded and current". It also flags a scan or industry file behind the MRS.
- **Pedal:**
  - Wilder ATR(14), the same as the scanner and the tickets.
  - The NORMAL line adds a computed sentence on today's downside against all days.
  - Wording made impersonal.
- **Scanner:** download retry; coverage guard (the previous scan is kept below 70% coverage, with a flag); an empty scan never overwrites.
- **Workflows:** the daily update pulls before pushing; the scanner commits the coverage flag.
- **Wording:** market statements made impersonal (regime lines, insurance, effective bets, avoid list).

## 6.3.1 · 2026-11-13

- **Market context card:** shows the cell for today's exact state (MRS side × SPY phase) with the strongest cell for scale. Sizing removed (owned by the pedal). One SPY-vs-50-day number (the pedal's).
- **Pedal:** REBOUND WINDOW renamed DIP ZONE; "your planned setups" wording.
- **Follow-through day label:** date only (the scan can be a session behind).
- **Allocation footer:** the tickets use ATR(10), as tested; stock setups use ATR(14).
- **Daily update:** NYSE trading calendar (Columbus / Veterans Day no longer skipped).
- **Fix:** the card no longer prints the percentile flag as the regime.

## 6.3 · 2026-11-13

- **New Market context card:** SPY phase track (dip zone → recovery → trend → out), what paid in the state, where to look, trade management.
- **Scanner:** grey-out of setups with a stop wider than 10%; "chasing" tag on breakouts in the dip zone or recovery; context line above the long scans (replaces the 10 Nov guide).
- **Market label:** follow-through day low status and odds.
- Removed: the "What it means for stock entries" line in the allocation card.

## 6.2.x · 2026-11-10 to 11-12

- **Allocation rule card** (SPY / QQQ rebound and breakout tickets, forward tracking).
- **Market label v5** (plain states).
- **Deep dip:** stop = trigger close − 2 ATR(14) (12 Nov; was ATR(10)).

## 6.0–6.1 · 2026-11-07 to 11-09

- **Scanner v3:**
  - fixed scans (All, Bases, Pullbacks, Deep dip, Hot 50 and the short mirrors);
  - liquidity filter;
  - Deep dip trigger scan;
  - setup chips;
  - RVOL column.
- **User guide v6.0;** evidence tags re-tested (audit v2).

## 5.x · 2026-10-30 to 11-06

- **Risk today panel** (pedal, today's state, effective bets, insurance cost, vol guide);
- **Avoid list;**
- **watchlist** on the `watchlist` branch;
- **bases and pivots;**
- **follow-through day state;**
- **band-confirming line;**
- **freshness flags;**
- **Friday-row fix.**

## 4.x and earlier · to 2026-10-29

- **Scoring MRS v4.0** (Sept 2026, Calibration Report v8.0);
- **branding (ATN);**
- **sector rotation map;**
- **industry drill-down;**
- **scanner and chart panel;**
- **Gamma Live zero gamma shadow.**
