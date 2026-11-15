# Changelog

Versions follow `docs/PROCESS.md`:
- MAJOR: scoring model or headline meaning;
- MINOR: new card, scan, rule or message;
- PATCH: fixes, wording, reliability.

The current version is in `VERSION` and in the page footer. Entries before 6.3.1 were reconstructed from the commit history and the project notes on 13 Nov 2026.

## 8.2.0 · 2026-11-15

The retired MRS score is off the page, and **Risk today** is rebuilt on the v8 state and shown open under the MRS v8 card.

- **Risk today** (`risk_panel.py`), four visual tiles:
  - **SPY vs its 50-day:** the ATR gauge and the sell-off episode day. In the dip zone it adds the tested wide-both-ways numbers (5%+ up 31% vs 17%, 7%+ drop 27% vs 14%).
  - **This state:** SPY's next-20-session distribution since 2008 for today's SPY phase × VIX width (P10–P90 bar, ≥5% up / down, fast drop of 7%), against all days. Descriptive.
  - **Clock:** C1 per-trade quality by sessions since the last DIP ZONE day (mean R and P(≥3R)), with today's bucket lit. It is per trade, not size.
  - **Effective bets:** unchanged (Tested).
  - **Removed from display:**
    - the pedal (its NORMAL / DIP ZONE / EASE OFF names clashed with the v8 stance, and EASE OFF needed the retired RISK-OFF band);
    - the MRS-band cell;
    - the Insurance tile (the hedge is on the card);
    - the volatility-target guide (not part of the v8 framework).
  - The nightly forward-log columns are still written.
- **Since last session:** "MRS −3.77 → −4.54" is replaced by the v8 stance and clock ("MRS v8 EASE (unchanged) · clock EARLY 9"). The rotation and deep-state lines, both keyed to the retired band, are removed.
- **The MRS v6 score card is no longer drawn.** The bottom diagnostics toggle is renamed "Retired MRS v6 score … (reference only)" and is off.
- **Laggards note:** "outside RISK-OFF" now reads "outside the dip window", with the note that the test used the old band.
- New `test_risk_panel_v8.py`: 3 checks.

## 8.1.0 · 2026-11-15

The scanner and the leader and laggard lists now read the v8 market state (new `state_view.py`): colour and chips, nothing hidden or locked.

- **Scan strip** above the long scans: one chip per scan showing how it did in today's SPY state (▲ works here · ● no edge · ▼ lagged a random entry). Hover shows the evidence (S2 survivorship-free, RS1). Examples:
  - with ticket A open, breakouts ▼ and deep dips and pullbacks ▲;
  - with no ticket, bases and breakouts ▼;
  - in the dip window, momentum ▼.
- **RS colours by state** (RS label and RS rating, scanner and watchlist panel):
  - flipped in the dip window, when SPY is in the DIP ZONE or ≤ 20 sessions after it (teal = laggard, copper = leader). RS1: leaders did worst there;
  - grey in a calm trend (RS1-a: no difference);
  - as before otherwise.
- **Momentum light** (leaders list): it now uses the v8 dip window instead of the retired MRS RISK-OFF band.
- **Laggards list:** in the dip window it is shown dimmed under "rebound window, not an avoid list today"; otherwise it is "Avoid … outside the dip window".
- **Heat score:** muted chip, and the lookup chips are labelled "describes, no edge" (top-quartile heat earned the same as the universe).
- **FTD chip and lookup chip:** grey, labelled commentary (C2).
- **Short scans fold:** relabelled "tested (SB1): avoid in the dip window". It is not locked.
- Already in place and unchanged: the breakout 5-day-low stop (+0.06R, tested); deep dips with a stop wider than 10% greyed (S1); the "chasing" tag.
- New `test_state_view.py`: 5 checks.

## 8.0.0 · 2026-11-15

**MAJOR: the exposure driver changes.** The new **MRS v8 card** (`exposure_guide.py`) sits on top of the page. It replaces the composite MRS score as the guide for exposure. The card is visual; the words are in a fold under it, "What each part of the card means", closed by default. The v6 score card (Today card) moves into a fold, "retiring", for comparison until mid-December. Version 7.0.0 was built locally but never shipped; it is folded into this release. Evidence: `Studies/Index_Edge_Program_2026-11/MRS_v7/` (C1, SB1, C2, L3 chain check, v8 sizing) and `Studies/00_GOAL_anchor.md`.

- **Stance badge** (PRESS / NORMAL / EASE), set by the SPY phase:
  - PRESS: ticket A or B open, 0.75% a trade, swing open risk ≤ 4%;
  - NORMAL: DIP ZONE, 0.75% a trade, ≤ 3%;
  - EASE: no ticket, 0.375% a trade, ≤ 1.5%, swing gross ≤ 60%.
  - v8 sizing run, 500 paths: this beat a flat 3% cap (CAGR 9.1% vs 7.5%, max drawdown −25.9% vs −28.8%) and every clock-based budget.
- **Clock tile:** sessions since SPY's last DIP ZONE day (DIP / 1–20 / 21–60 / 60+). It is per-trade context (C1, in sample), not size.
- **SPY phase pills** and a **VIX width gauge** against the median since 1990 (17.56).
- **SPY tickets:** 1.5% risk each (Fabio, 15 Nov 2026), with unit size and stop or the buy level.
- **Exit ladder:** stop → +2R break-even → trail the 21 EMA → no cap.
- **Where to look:** laggards and dips in the dip and the early window; "no leader edge" later.
- **Hedge tile:**
  - a window shows only in the 5 sessions after a trend start or a trend end (the 60-session ~5% OTM put);
  - "never roll puts";
  - the real put price comes from the new daily put log.
- **Footer:** sessions since the dip; last follow-through day (commentary only).
- **New `put_log.py`:** the daily workflow logs SPY put quotes into `mrs_history.csv` (`pl_*` columns): 20 and 60 sessions, ATM / 5% / 10% OTM, mid, implied vol, model price. It needs no workflow change. Until the first logged session, the card shows the 13 Nov IBKR quote as a seed.
- **`risk_panel.py`:** the modelled "Insurance cost" tile is retired. It priced the hedge about 25% too cheap and advised rolling. `put3m_cost` is still logged.
- **Chart (`chart_view.py`):** back to Fabio's original lines: EMA 21 light teal, SMA 50 bright grey, SMA 200 orange, SMA 250 dashed orange. Candles are teal up / orange down; the SPY phase colours and the phase key are removed. The ATR lines and markers stay.
- **Not on the card by decision:** the composite score, the short book, the follow-through day as a rule, RS tilt, margin.
- `test_exposure_guide.py`: 6 checks (clock, stance, hedge window, budget, card render, put-log row).

## 6.5.0 · 2026-11-14

The daily chart (scanner, watchlist and ticker lookup) is drawn like the TradingView "Ext·RS" indicator v7.5 with Fabio's settings (`chart_view.py`).

- Candles in the SPY phase colour, from the same allocation replay as the Today card: Dip zone, Recovery, Trend, Out. Hollow on up days, filled on down days.
- SMA 50, SMA 200 and SMA 250 (dashed). The EMA 10 is removed; the EMA 21, the 52-week high and the 21-day high are in the legend (click to show).
- Dotted SMA 50 ± 2 to 5 ATR(14) lines over the last 20 sessions, labelled at the right edge (one legend entry toggles them).
- Markers: Deep dip trigger (grey when the 2 ATR stop is wider than 10%), breakout while SPY is in Dip zone / Recovery (stocks only), 50-day reclaim after RSI(14) < 35 (R1).
- Unchanged: 10/10 swing price labels, the base overlay, the volume pane, the zoom that carries over between tickers.
- Checked on the EODHD panel (PBR, NVDA, XLE): renders; Yahoo is blocked on the device, so not run inside the app here.

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
