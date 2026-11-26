# Changelog

Versions follow `docs/PROCESS.md`:
- MAJOR: scoring model or headline meaning;
- MINOR: new card, scan, rule or message;
- PATCH: fixes, wording, reliability.

The current version is in `VERSION` and in the page footer. Entries before 6.3.1 were reconstructed from the commit history and the project notes on 13 Nov 2026.

## 8.13.0 · 2026-11-26
- **Hollow candles** (TradingView style) on every chart: the stock chart panel (scanner, lookup, watchlist) and the SPY chart on the stance card. Colour = close against the prior close (teal up, orange down); the body is hollow when the close is at or above the open and filled when below. No levels, lines or markers changed.
- **Export to TradingView.** Under the scanner table, a button downloads the list as shown (the selected scan with the theme filter applied, in the table's order) as a TradingView watchlist file: one section named after the scan and the date. The watchlist has the same button next to Save / Discard (names ticked Remove are left out). In TradingView: watchlist menu → Import list. Class shares are written the TradingView way (BRK-B → BRK.B). Logic in `tv_export.py`, tests in `test_tv_export_8130.py`.

## 8.12.1 · 2026-11-25
- **Stance card, "What dips did next":** the middle row now shows today's case on its own when SPY is below a still-rising 200-day: **Line lost** (30 S&P 500 dips since 1963): another 3% within 20 sessions 47% (was 44% pooled), 5% 37%, 7.5% 13%, 10% 7%; another 10% within 60 sessions 23%. The old pooled row (34 dips) included 4 dips above a 200-day that was not rising; it is shown only when that is today's case, labelled as pooled. The label line under the badge and the next-close sentence use the same numbers. Descriptive (DF1b split).
- **Leading industries and laggards wording:** the stock-level result (beaten-down names beat prior leaders) is now stated for the DIP ZONE itself, where it was tested and confirmed out of sample (RS1-b, 2001–08); over the 20 sessions after the zone the gap was noise. In CAUTION the "where to look" line now points to beaten-down names in the DIP ZONE instead of the structural leaders; the leaders' replay result stays as context. No list, filter or condition changed.

## 8.12.0 · 2026-11-25
- **Theme filter on the scanner, every scan.** A row under the scan pills narrows the list to a market theme:
  - **Sector:** one or more sectors.
  - **Industry group:** one or more NASDAQ industry groups (type to search). With sectors chosen, only their groups are listed.
  - **Leading sectors:** sector RS score +1 or higher.
  - **Industry role:** industry leader, leader in a leading industry, or any stock in a leading industry.
- Each option shows how many stocks of the current scan sit in it. When a theme is on, a THEME line above the table says what is applied and how many stocks remain (e.g. 4 of 37). Parts combine with AND; several sectors or groups combine with OR.
- Brings back the sector and industry filters removed on 7 Nov 2026, now with several sectors and groups at once. Descriptive: a theme narrows where to look; it is not a tested edge. Logic in `theme_filter.py`, tests in `test_theme_filter.py`.

## 8.11.0 · 2026-11-22
- **Entry timing tile replaces the Hedge tile** (bottom row of the stance card). It reads "signal close or next open" on ordinary days. When the week's last session is a **weak Friday** (SPY's move in the bottom 10% of all daily moves, in ATR), it shows WEAK FRIDAY with the move and "long entries → Monday's open" until Monday's session starts. Evidence (WD1, tested): after a weak Friday, Monday opened 0.5 ATR or more lower 18% of the time vs 8% after other Fridays (SPY 1995–2026; S&P 500 checks since 1962), −0.11 ATR on average, both eras, same VIX mix; the median gap is still slightly up and Monday's session recovers part of it.
- **No standing hedge rule:** the ~60-session ~5% OTM put is off the card. The account test (CA1, 22 Nov) found it cost return in 2009–26 and deepened the drawdown in both eras. A grey line on the tile says so; the fold explains it. The daily put log keeps running as data.
- Weeks that end on a Thursday (Good Friday and other exchange holidays) are handled: the rule looks at the week's last session.

## 8.10.3 · 2026-11-21
- **Stance card evidence tag:** Clock · width · budget now reads TESTED · selection-adjusted (was TESTED · v8 sizing). DS1 (21 Nov) checked that the budget's edge over a flat 3% cap survives having been chosen from 15 tested budgets (Hansen SPA / Romano–Wolf, 2009–26; same direction 2001–08). No numbers or rules changed.

## 8.10.2 · 2026-11-21
- **Stance card layout (wide screens):** the SPY chart takes less width (0.56 : 1.44 instead of 0.8 : 1.35) and is taller, so it fills its tile. The Clock · width · budget tile gets the width: larger rings, larger text in the 200-day filter line, the dip-odds table, the distance-to-the-next-stance axis and the swing budget bars. The bottom row (SPY phase · tickets, Exit, Hedge) has larger chips, pills and text. Below 1,900 px the card keeps the previous split, so laptops are unchanged in layout. No numbers or wording changed.

## 8.10.1 · 2026-11-21
- **Joint view, filter tab:** one line in the reading says how it differs from Risk today: the chart is the worst point reached within 60 sessions (downside only), not where SPY ends; Risk today shows where it ended after 20 sessions, both ways.

## 8.10.0 · 2026-11-21
- **Joint view card (Descriptive; the filter split itself Tested).** Two joint plots under the Vol surface, each with today's point and a plain-language reading beside it:
  - **200-day filter:** S&P 500 DIP ZONE days since 1963, dip depth (ATR below the 50-day) against the worst further fall within 60 sessions, filter ON vs OFF. Today's depth is marked on today's panel with the median and worst-10% further fall of the 120 nearest days.
  - **Fear vs price:** S&P 500 daily return against the change in VIX since 1990, VIX CALM vs today's width. Today's point, the typical slope, and where today's VIX change sits among days with a similar SPY move (less fear / normal / more fear than usual).
- Brand colours: teal = filter ON / CALM, orange = filter OFF / STRESSED, copper = WIDE. A fold explains how to read a joint plot.
- History in `joint_long.csv`, built offline by `build_joint_long.py` (re-run once or twice a year, like `risk_long.csv`).

## 8.9.0 · 2026-11-20
- **Vol surface charts carry their own reading (Descriptive, stage 1).** Smile: the spot, the next-expiry move band (± the ATM straddle), the 30-day 25-delta put and call marked on the smile, and a box with the 30-day RR25 (call − put IV) and the −5% put skew over ATM. Term structure: the next and 30-day ATM IV labelled, a box with the front − 30d slope and its reading (inverted / flat / upward), the VIX close as a dotted line, and the event hump with its forward vol. The chart's download button gives the annotated image for the note. All from the snapshot the panel already reads; Gamma Live levels (walls, flips) are not on it (stage 2).
- **Morning read:** the term line now states the slope as front − 30d (positive = inverted), the same sign as the chart and the Gamma Live vol log.

## 8.8.0 · 2026-11-19
- **Risk label (Descriptive).** A plain-language risk-on / risk-off line under the stance badge, computed from the same two inputs as the stance (SPY phase and the 200-day filter), so the two never disagree: ticket B → RISK-ON · TREND; ticket A → RISK-ON · RECOVERY (either below a falling 200-day → NEUTRAL · BEAR RALLY); DIP ZONE → RISK-OFF · BUYABLE DIP (above a rising 200-day), LINE LOST (below a still-rising one), LINE NOT RISING, or DOWNTREND (below a falling one); no ticket → NEUTRAL · NO TICKET (below a falling 200-day → RISK-OFF · DOWNTREND). The VIX band sits beside it as width and never changes it. The stance alone sets the size: a buyable dip is risk-off in market terms and NORMAL size.
- **Scanner banner** starts with the same label (it showed the retired score's band before).
- **Explainer fold:** the full state → label table.

## 8.7.2 · 2026-11-19
- **Scanner:** all 21 columns are shown by default; the "More columns" switch is removed (Fabio). The ticker column stays fixed on the left while the table scrolls sideways.

## 8.7.1 · 2026-11-18
- **VIX median updates itself.** The line between CALM, WIDE and STRESSED is the median VIX close since 1990. It was a fixed number (17.56, measured on 13 Nov 2026); it is now recomputed at every run from `vix_1990.csv` (VIX closes 1990-01-03 to 2026-11-13) plus every later close in `mrs_history.csv`, the same expanding median the research uses. If the file cannot be read, the last value (17.56) stays. The card, the explainer fold ("median on <date>") and Risk today all read the same value. It moves by hundredths a year; `build_risk_long.py` (offline, Risk today history) keeps 17.56.

## 8.7.0 · 2026-11-18
- **"What dips did next, by the 200-day"** under the 200-day filter on the MRS v8 card (Descriptive · SPX dips 1963–2026, DF1b): the share of S&P 500 dips that fell a further 3 / 5 / 7.5 / 10% within 20 sessions (and 10% within 60), for each 200-day state, today's row highlighted. Above a rising 200-day: 36 / 17 / 8 / 4% (any day 39 / 20 / 9 / 5%); below a falling one: 67 / 44 / 29 / 16%. Outside the DIP ZONE the line reads "if SPY dips into the zone in this state".
- **Next-close level for the filter:** the SPY close needed next session to keep (or turn) the filter ON, both conditions included (close above tomorrow's 200-day, and that 200-day above its value 20 sessions earlier). One close decides (FB1: no buffer passed).

## 8.6.0 · 2026-11-17
- **Risk today, longer history.** "This state" now uses the S&P 500 since 1990 (906 days, 83 dips for the DIP ZONE with the VIX WIDE) instead of SPY since 2008; the numbers held in 1990–2007 and 2008–2026. Data in `risk_long.csv`, built offline by `build_risk_long.py` (re-run once or twice a year); without the file the panel falls back to SPY since 2008.
- **Wording fixed:** "≥5% up / down" is where the index *ended* after 20 sessions ("ended ≥5% up"); only the fast 7% drop counts any point in the 20 sessions.
- **Dip-zone line** under "SPY vs its 50-day" is computed from the same history (was a fixed SPY-2008 sentence).
- **Clock table:** random stock entries 2001–26 (both eras) instead of 2009–26 in sample, split by the 200-day filter. Filter ON: dip +0.30R, 1–20 +0.17R, 21–60 +0.10R, 60+ −0.06R (same sign in both eras). Filter OFF: the dip is not an edge.
- **Deep dip scan text:** the retired MILD RISK-OFF / RISK-OFF evidence replaced by the 2001–26 result (setups frame the trade; the state carries it).

## 8.5.4 · 2026-11-17

- **Sidebar:** the Daily Inputs form (B20%, ADL, zero gamma, PC ratio, Submit & Update) is removed. Those inputs fed only the retired composite MRS score; the automated nightly run keeps filling what it can, and the diagnostics tabs show the last values.
- **Sidebar data note:** freshness is now measured against the last completed NYSE session, the same rule as the data check strip. Before the US open it read "1 session behind" with current data.
- **Vol surface:** 16 Nov added to `vol_gl.json` from the Gamma Live close (the automatic push needs the `vol_snapshot.yml` workflow on GitHub).

## 8.5.3 · 2026-11-16

While the US session is open, the stance badge shows the stance at the **last close**, and a dashed label beside it shows the live reading, marked provisional: e.g. "Mon 16 Nov · live: NORMAL if SPY closes below 752.91 (now 744.98) · provisional". After 16:05 New York time the label disappears and the badge shows the day's stance. The rings, filter line, levels and budget bars keep showing the live bar. (Fabio, 16 Nov: the badge switched from EASE to NORMAL intraday while the header still showed Friday's date.)

## 8.5.2 · 2026-11-16

Layout (Fabio: smaller SPY chart, more room for the rings and visuals).
- MRS v8 card: the SPY chart column narrows (about 37% of the width instead of 47%); the clock / width / budget column widens.
- The "Distance to the next stance" axis is capped at 980 px wide, so its labels stay about 12 px on a 2560 px screen.

## 8.5.1 · 2026-11-16

Wording (Fabio: the setups still frame discretionary trades when the state is conducive).
- **Scanner strip:** the chip says whether the **state paid** for that scan's setups (▲ the state paid · ● about flat · ▼ lagged · ○ untested), not "works / no edge". Hover text adds: use it to frame entry, stop and target; the return came from the state. In the DIP ZONE the deep dip and pullback chips follow the 200-day filter (ON: ▲; OFF: ●, dips kept falling).
- **Card, Where to look** outside the dip zone: "setups frame the trade · the state carries it" (was "no pattern beats random").
- **Explainer:** the setups find clean structures and set entry, stop and target; the state decides whether to trade and how big.

## 8.5.0 · 2026-11-16

Rule change after the out-of-sample checks (2001–08 on the survivorship-free 2000 panel, and the S&P 500 since 1963). Notes: `claude/mrs_oos_2000_2008_findings.md`, `claude/mrs_df1_spx_dip_filter_findings.md`.
- **New 200-day filter** on the MRS v8 card: ON when SPY closes above its 200-day and the 200-day is higher than 20 sessions ago. On the S&P 500 since 1963, dips below a falling 200-day fell another 10% within 60 sessions 45% of the time, against 12% above a rising one.
- **Stance:** a DIP ZONE keeps NORMAL (0.75% a trade, ≤ 3%) only when the filter is ON. Otherwise it is EASE (0.375%, ≤ 1.5%, gross ≤ 60%). PRESS and EASE are unchanged. In the account simulation this beat the previous v8 budget in 2009–26 (CAGR 9.6% vs 9.1%, max drawdown −22.7% vs −25.9%) and in 2001–08 (−1.1% vs −2.9%, −31.9% vs −39.3%).
- **Clock:** reads with the filter. Above a rising 200-day, early entries were the best and late ones the worst in both periods; below a falling 200-day the early ring turns orange (the dip was not an edge in 2001–08).
- **Where to look:** in the DIP ZONE, beaten-down names (bottom 40% by 6-month return) rather than prior leaders, tested on 2001–08. Outside the zone: no RS edge, and no chart pattern beats a random entry. "Deep dips" is no longer a place to look.
- **Scanner strip and RS colours:** deep dip and pullback scans now read "no edge" (no better than a random entry in the same state, 2001–26). The flipped RS colours apply in the DIP ZONE only (the 20 sessions after it showed no RS gap out of sample).

## 8.4.1 · 2026-11-16

Layout pass on the live page (Fabio: smaller SPY chart, less unused black space, more room for the data visuals). Same numbers.
- **MRS v8 card:**
  - SPY chart: 120 sessions instead of 160, in a narrower tile (about half the card).
  - The right tile fills the space with two new visuals under compact rings:
    - **Distance to the next stance:** one price axis with SPY against the levels that change the stance (DIP ZONE line → NORMAL, 50-day / A unlock, B trigger → PRESS, open-ticket stops), each with its % distance;
    - **Swing budget by stance:** PRESS / NORMAL / EASE as bars (cap on a 0–4% scale), today's highlighted.
- **Risk today:**
  - "SPY vs its 50-day" and "Effective bets" are stacked in one column.
  - The distribution and the clock tiles use the full height.
  - New 60-session sparkline of SPY's distance from its 50-day in ATR, with the DIP ZONE shaded.

## 8.4.0 · 2026-11-16

Design audit: one ATN component set for the whole page, plus a new Vol surface panel (Fabio's picks: card B with candles + A's detail row; Risk today B curve + A bars; Gamma Live + Yahoo backup). Notes: `claude/mrs_design_8_4.md`.

- **One component set** (new `ui.py`, injected once):
  - tags: UPPERCASE 9.5px everywhere. TESTED copper fill, PROVISIONAL copper dashed, DESCRIPTIVE grey, MODELLED light-copper dashed; any qualifier in grey after it;
  - bars: one 10px track with a white pin and triangle head (the card's 12px and Risk today's 16px bars are gone);
  - tiles, labels, kickers: one size and spacing; numbers in IBM Plex Mono;
  - colours: ATN tokens only. The non-brand reds (#e07a5f, #7a3a2e, #ff4f5e) and the second muted teal are replaced; adverse = orange;
  - older cards (allocation, leaders, Market detail) are restyled through the same CSS, without changing their content.
- **MRS v8 card redrawn** (`exposure_guide.card_v9_html`):
  - stance badge and budget on top;
  - SPY candles of the last 160 sessions with the phase as background bands, and a dotted "today · clock" line;
  - three rings: clock, VIX, open-risk cap;
  - one detail row: SPY phase and tickets · exit and where to look · hedge;
  - the explainer fold (closed) explains the picture. `card_html` is kept for reference.
- **Risk today redrawn** (`risk_panel.panel_v9_html`): the "This state" tile is the distribution itself (SPY next 20 sessions, this state vs all days, P10 / P90 marked) with three odds bars (≥ 5% up, ≥ 5% down, 7% fast drop) pinned at the all-days rate. Same numbers as 8.2.0.
- **Vol surface panel** (new `vol_core.py`, `vol_view.py`), under Risk today, open. Descriptive.
  - **Content:** SPY smile (next, Friday, ~30-day expiries; dashed = the session before) and ATM term structure.
  - **Morning read:** next-expiry straddle and event premium, term slope, 30-day skew, VIX vs 30-day ATM, event hump.
  - **Maths:** ported unchanged from Gamma Live panels.py v5.9 (checked equal on 12 and 13 Nov).
  - **Sources:**
    - Gamma Live pushes its close snapshot after 16:15 ET (`mrs_push.py` v5.9.14 → new workflow `vol_snapshot.yml` → `vol_snapshot.py` → `vol_gl.json`);
    - the nightly update writes a Yahoo backup (`vol_yahoo.json`);
    - Gamma Live wins per date, and the panel names its source.
  - **Data check:** new "Vol surface" chip.
- Tests: `test_design_84.py`.

## 8.3.2 · 2026-11-15

- The Market detail fold is open by default; the Watchlist table and chart is folded by default (Fabio).

## 8.3.1 · 2026-11-15

Market detail fold brought in line with v8:
- **Allocation card footer:** "100% of the allocated capital while a ticket is open" is replaced by the v8 sizing (one unit per ticket risking 1.5% of NLV at its stop).
- **Leaders list in the dip window (Momentum CAUTION):**
  - the title reads "Leading industries — context in the dip window, not a buy list";
  - the text follows RS1 / R1 instead of the unconfirmed "+0.6 pp industry rebound";
  - "size by the swing-size line" (a retired MRS-band line) becomes "size by the MRS v8 card's swing budget".

## 8.3.0 · 2026-11-15

- **Morning data check** (new `data_check.py`): one strip on top of the page.
  - **What it checks:** every automatic source against the last completed NYSE session (New York close, NYSE holiday calendar): daily update, VIX, put log, SPY prices, SPY tickets, Risk today inputs, sectors, industries, scanner.
  - **Chip status:**
    - ✓ teal = current;
    - ! copper = one session late;
    - ✕ orange = older or missing;
    - · grey = put log waiting for its first nightly run.
  - **Header:** "DATA · ALL CURRENT", "n LATE" or "n ISSUES". The load-health issues list follows it; it replaces the v6.3.2 health line.
- **Chart (`chart_view.py`), as in the TradingView Ext·RS v8.0 indicator:**
  - **ATR zones:** the ATR(14) zones from the SMA 50 are faint shading under the candles (below −1/−2, −2/−3, −3/−5 teal, darker further down; above +2/+3, +3/+5 copper), with labels at the right edge. The dotted ATR lines are gone.
  - **21-day high:** dotted over the last 60 sessions, with a label.
- **Laggards banner (dip window):** it no longer cites the +0.7 pp industry rebound. R1 found laggard *industries* did not beat leading ones after beta; the banner now points to laggard *stocks* (RS1).
- New `test_data_check.py`: 2 checks.

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
