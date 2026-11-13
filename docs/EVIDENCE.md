# Evidence register

Every claim, rule or number the dashboard shows, the evidence behind it, and its status. When a claim is added, changed or retired, this file changes in the same release (see PROCESS.md, step 5).

**Status tags** (as on screen):

| Tag | Meaning |
|---|---|
| **TESTED** | Pre-specified test, non-overlapping samples, bootstrap CI, survived the multiple-testing correction, same sign in both halves. |
| **PROVISIONAL** | Clear in-sample result that failed one of the TESTED gates, or is exploratory. Used at reduced size; logged forward. |
| **DESCRIPTIVE** | A measured state or definition, not a tested edge. |
| **NULL** | Tested, no effect. Kept on screen only as a tool, not as a reason to trade. |

**Sources:** project notes are `claude/<name>.md` in the Epistruct project. Study folders are under `TRADING DESK/Studies/` or `studies/` in the Epistruct folder.

## Score and regime

| ID | Where | Claim | Status | Source |
|---|---|---|---|---|
| E-01 | Decision strip | The MRS is a dispersion dial: it measures how wide the next weeks can be, not which way; RISK-OFF has not had a worse average return than RISK-ON. | TESTED | MRS Calibration Report v8.0 (Sept 2026); `README.md`; `claude/project_mrs_dashboard` |
| E-02 | What to do (band lines) | Swing width by band; the stop stays where the setup is wrong; RISK-OFF is not a sell signal. | TESTED | Calibration Report v8.0; `claude/mrs_evidence_audit_v2.md` |
| E-03 | DEEP RISK-OFF line | 10-session drop of 3%+ on 31% of days vs 19%; not a bottom call. | PROVISIONAL | `claude/mrs_evidence_audit_v2.md` (mrs_risk_control) |
| E-04 | Rotation line | Defensives no edge in RISK-OFF; commodities and metals +0.6 pp vs SPY at 20 sessions, +2 to +3 pp at 60–90. | PROVISIONAL | `claude/industry_by_mrs_band_v0.md` |
| E-05 | Market label (SPY / QQQ) | Correction, rally attempt, follow-through day, range, above / below, uptrend / pullback: definitions. | DESCRIPTIVE | `claude/mrs_market_label_v5_2026-11-11.md` |
| E-06 | Market label, follow-through day low | Undercut within 4 sessions → the rally failed within 20 sessions 42% of the time vs 16% when the low held; RVOL for information only (no reliable difference). | PROVISIONAL (exploratory) | FTD study, 12 Nov 2026 (`/home/claude/ftd2`, SPY 1994–2026 + QQQ 1999–2026, 164 events); `claude/mrs_v6_3_market_context_2026-11-13.md` |

## Risk today

| ID | Where | Claim | Status | Source |
|---|---|---|---|---|
| E-10 | Pedal DIP ZONE | SPY > 1 ATR(14) below its 50-day: wide both ways, more than the VIX implies (5%+ up 31% vs 17% at the same VIX; 7%+ drop 27% vs 14%). | TESTED (width) | `claude/mrs_spec_v5_risk_panel.md`, `claude/mrs_risk_control_step5.md`; Wilder re-check 13 Nov 2026 (`docs/audits/AUDIT_2026-11-13.md`, C3) |
| E-11 | Pedal EASE OFF | RISK-OFF and SPY below its level of a year ago: fast drops of 7%+ in 1 of 3 cases (all days 1 in 8). | TESTED | same as E-10 |
| E-12 | Pedal NORMAL | Normal process; the tail sentence is computed live from today's state. | DESCRIPTIVE | `risk_panel.normal_tails` |
| E-13 | Today's state | P10 / median / P90 and tail rates of SPY's next 20 sessions in the same state since May 2008. | DESCRIPTIVE | `risk_panel.spy_states` (live) |
| E-14 | Effective bets | Trailing 63-day industry correlation tracks the next month's (Spearman 0.53, both halves); 5 ÷ (1 + 4ρ); about 1.5 bets in stress. | TESTED | `claude/mrs_spec_v5_risk_panel.md` |
| E-15 | Insurance cost | Puts in RISK-OFF with SPY 12-month up returned 5% (10% OTM) to 23% (ATM) of their cost. | PROVISIONAL | `claude/mrs_dashboard_user_guide_v6_0.md` (risk-control studies) |
| E-16 | Book volatility guide | Weekly vol scaling cut the max drawdown by about 12 pp at a similar Sharpe. | PROVISIONAL | `claude/mrs_spec_v5_risk_panel.md` |

## Market context card and allocation

| ID | Where | Claim | Status | Source |
|---|---|---|---|---|
| E-20 | Allocation tickets A / B | Rebound (first close ≤ −1 ATR below the 50-day) and 21-day breakout tickets; EITHER 10.2% CAGR, −31% max DD vs buy-and-hold 9.2%, −56% (SPY 1995–2026). | PROVISIONAL (in-sample) | `claude/entry_exit_combo_2026-11-10.md`, `claude/rebound_depth_v0_findings.md`, `claude/spy_breakout_vs_buyhold_2026-11-09.md`, `claude/mrs_v6_2_allocation_card.md` |
| E-21 | Context card, SPY phase | DIP ZONE / RECOVERY / TREND / OUT from the pedal distance and the tickets. | DESCRIPTIVE | `context_card.py`; `claude/mrs_v6_3_market_context_2026-11-13.md` |
| E-22 | Context card, "this state" cells | Mean R per trade by MRS side × SPY phase for Deep dip, 50-day pullback, breakout and random uptrend entries. | PROVISIONAL (exploratory: cells drawn after S2) | `claude/setups_state_s2_findings.md` |
| E-23 | Context card, setups | No setup beat a random uptrend entry in the same SPY allocation state; breakouts were below random with the rebound ticket open (−0.06R). | TESTED (null / negative) | `claude/setups_state_s2_findings.md` |
| E-24 | Trade management line | Stop 2 ATR(14), max 10%; 1.5 ATR costs about 0.05R; take 2R, then trail the 21 EMA (+0.03 to +0.08R over a plain 2R). | TESTED (stops, S1 / S2) · PROVISIONAL (trail) | `claude/deep_dip_stop_width_findings.md`, `claude/setups_state_s2_findings.md` |

## Scanner

| ID | Where | Claim | Status | Source |
|---|---|---|---|---|
| E-30 | Deep dip scan and stop | Trigger rules; stop = trigger close − 2 ATR(14); MILD RISK-OFF / RISK-OFF +0.09R over random in the same band. | PROVISIONAL | `claude/deep_dip_v3_stocks_findings.md`, `claude/deep_dip_stop_check_2026-11-09.md`, `claude/mrs_deep_dip_atr14_2026-11-12.md` |
| E-31 | Grey-out (stop > 10%) | Deep dips with a stop wider than 10% did worse than the rest and than random entries in equally volatile stocks (21 EMA exit). | TESTED (S1, primary exit) | `claude/deep_dip_stop_width_findings.md` |
| E-32 | "chasing" tag on breakouts | Breakouts below random with the rebound ticket open; lagged in the dip zone and recovery. | TESTED (rebound state) · PROVISIONAL (phases) | `claude/setups_state_s2_findings.md` |
| E-33 | Base / pivot / breakout patterns | Pattern definitions; mechanical base breakout did not beat random entries. | DESCRIPTIVE · NULL (edge) | `claude/setup_study_B_base_breakout_findings.md`, `claude/setup_study_v0_findings.md` |
| E-34 | Pullback (21 EMA) | No edge over random entries in the same trend; better in MILD RISK-OFF / RISK-OFF. | NULL · PROVISIONAL (regime) | `claude/pullback_21ema_v0_findings.md`, S2 |
| E-35 | Extension zones (Above / Far above) | The left tail did not grow with extension in stocks. | TESTED (no growth) | `claude/stretched_label_v0.md` |
| E-36 | Heat score | Sum of displayed states; not a forecast. | DESCRIPTIVE · NULL (as forecast) | `claude/mrs_dashboard_user_guide_v6_0.md` |
| E-37 | Avoid list | Not for shorting in RISK-OFF: laggard shorts lost 2.7% a month, ≥5% squeeze in 56% of months. | TESTED | `claude/short_list_and_model_portfolio_v0.md` |

## Open items

- **E-06:** the FTD study has no project note of its own yet. Write `claude/ftd_low_study_2026-11-12.md` with the method and the tables.
- All context-card cells (E-22) and the trail exit (E-24) are logged forward from 13 Nov 2026. Review after about 6 months of live data.
- Survivorship: every stock-level result uses today's universe. A rerun on survivorship-free data is planned.
