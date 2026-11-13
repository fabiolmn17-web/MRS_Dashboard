# Decision log

One entry per decision that changes a rule, a threshold, a model, a data source or what the user is told. Newest first. Format: date · decision · why · evidence · version.

## 2026-11-13

- **Pedal ATR(14) moved to Wilder smoothing** (was a simple 14-day mean).
  - Why: one ATR definition everywhere (scanner, allocation tickets, indicator). Near −1 ATR the two could disagree on the dip zone.
  - Evidence: SPY 2008–26 re-check. Dip-zone days 948 vs 940; 5%+ up 28% vs 29%; 7% fast drops 27% vs 27%. The pedal statistics stand.
  - Version: v6.3.2.
- **Load-health line under the decision strip** (failed or stale sections listed with the reason).
  - Why: about 20 silent failure handlers; a missing card was indistinguishable from "no data".
  - Version: v6.3.2.
- **Scanner coverage guard and download retry.** A scan covering under 70% of the previous one is not published; the previous scan stays and a flag says why. An empty "no candidates" scan never overwrites.
  - Why: partial Yahoo downloads silently shrank the lists.
  - Version: v6.3.2.
- **Daily update pulls before pushing.**
  - Why: a collision with another commit lost the day's row.
  - Version: v6.3.2 (workflow).
- **NYSE trading calendar** (Columbus Day and Veterans Day are trading days; Good Friday by rule).
  - Why: the federal calendar skipped those sessions.
  - Version: v6.3.1.
- **One owner per question on the page.**
  - Sizing belongs to the pedal. The context card says what paid in today's exact state (regime side × SPY phase) and where to look.
  - Why: conflicting messages ("be active" vs "no edge" vs "smaller size").
  - Version: v6.3.1.
- **Dip zone = rebound window**, one name and one number on the page (the pedal's distance).
  - Version: v6.3.1.
- **Impersonal wording** in all market statements (no "you / your").
  - Why: owner's standard.
  - Version: v6.3.2.

## 2026-11-12 / 13

- **Market context card** (regime → SPY phase → setups → trade management) replaces the 10 Nov stock-entry guide.
  - Why: the S2 study. No setup beat random entries in the same state; the state carries the return.
  - Evidence: `claude/setups_state_s2_findings.md` (E-22, E-23).
  - Version: v6.3.
- **Scanner grey-out of setups with a stop wider than 10%**, and a **"chasing" tag** on breakouts in the dip zone or recovery.
  - Evidence: S1, S2 (E-31, E-32).
  - Version: v6.3.
- **Follow-through day low** in the market label: date, RVOL, undercut status, odds.
  - Evidence: FTD study (E-06).
  - Version: v6.3.
- **Deep dip stop on ATR(14)** (was ATR(10)); scanner and ExtMom v6.7 aligned. The allocation tickets keep ATR(10), as tested.
  - Evidence: 2 ATR(14) vs 2 ATR(10): −0.002R, no difference (S1 addendum).
  - Version: scanner 12 Nov.

## Earlier (reconstructed from the project notes)

- **2026-11-11 · Market label v5:** plain states (range / above / below / uptrend / pullback / correction / follow-through). `claude/mrs_market_label_v5_2026-11-11.md`.
- **2026-11-10 · Allocation card v6.2:** SPY / QQQ rebound and breakout tickets, forward tracking. `claude/mrs_v6_2_allocation_card.md`.
- **2026-11-09 · Deep dip stop = trigger close − 2 ATR(10)** (was trigger-day low − 1 ATR(10)); no meaningful difference. `claude/deep_dip_stop_check_2026-11-09.md`.
- **2026-10-30 · Risk today panel v5:** pedal, today's state, effective bets, insurance cost, vol guide. `claude/mrs_spec_v5_risk_panel.md`.
- **2026-09 · Scoring MRS v4.0** (Calibration Report v8.0, dispersion re-test).
