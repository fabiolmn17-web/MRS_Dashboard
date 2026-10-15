# MRS_Dashboard

Streamlit dashboard for the **Market Regime Score (MRS)** — Epistruct / Invariant Research.
Current scoring: **MRS v4.0** (Calibration Report v8.0, September 2026).

## What it measures

MRS is a **dispersion dial**: it estimates how far SPY tends to travel, in both directions, over the
next 5–60 sessions. Negative = wider two-sided swings, positive = calmer, narrower market. It does not
predict direction (Report v7.0: 0/29 timing tests survive FDR; 27/35 dispersion tests do).

Bands: RISK-ON ≥ 1.5 · MILD RISK-ON ≥ 0.5 · NEUTRAL ≥ −0.5 · MILD RISK-OFF ≥ −1.5 · RISK-OFF.

## Scoring (pipeline.py)

`MRS = Σ weight × state score` over nine components; each component's state comes from its 3-year
rolling percentile (Phi).

| Component | Weight | State scores |
|---|---|---|
| VIX | 2.22 | Continuous: `VIX_INTERCEPT − 3 × Phi` (0.906 − 3·Phi) |
| Extension | 0.91 | Compressed −0.5 · Normal +0.25 · Extended +0.5 |
| Momentum | 0.91 | Weak −0.5 · Normal +0.25 · Strong +0.25 (5-day % slope of SMA50) |
| ADL | 0.45 | Weak −0.5 · Normal +0.25 · Strong +0.25 |
| B20 | 0.45 | Low −0.5 · Normal +0.25 · High +0.25 |
| PC Ratio (10-day avg) | 0.91 | <0.944 +0.25 · 0.944–1.003 −0.25 · ≥1.003 −0.5 |
| SKEW | 0.91 | DANGER −0.5 · Low −0.25 · Mid 0 · High +0.25 · SAFE +0.5 |
| Zero Gamma | 1.00 | Above +0.5 · Near 0 · Below −0.5 |
| Volume Divergence | 0.00 | Information only |

Calibration: state score = −ln(large-move rate in state ÷ all sessions), 10D/21D, 2008–2026; VIX is
the anchor (~65%), the rest an equal-weighted context block. Out of sample never worse than v3.0 or
VIX alone. Full rationale: Report v8.0 and `research/studies/v8_component_retest` (outside the repo).

## Daily operation

- GitHub Actions `daily_update.yml` → `update_data.py`: fetch market data, append the day, rescore the
  full history with `pipeline.score_dataframe`.
- Manual inputs in the sidebar (B20%, ADL, Zero Gamma, PC Ratio): **typed values always win**; the
  automated run only fills blanks.
- After changing `pipeline.py`, reboot the app on Streamlit Cloud.

## Files

| File | Role |
|---|---|
| `app.py` | Dashboard UI |
| `pipeline.py` | Scoring engine (single source of truth) |
| `risk_dial.py` | Risk Dial band profiles, score percentile, stale-PC check |
| `auto_fetch.py`, `update_data.py`, `backfill.py` | Data fetch and updates |
| `apply_v4_rescore.py` | One-off migration to v4.0 (applied 2026-09-27) |
| `test_scoring_v4.py` | Tests: `python test_scoring_v4.py` |
| `mrs_history.csv` | Full scored history |

## Version history

| Version | Date | Change |
|---|---|---|
| v2.0 | Jul 2026 | Nine components, 5-study calibration |
| v3.0 | Sep 2026 | TRR-proportional weights, PC zones revised, Risk Dial dashboard (Report v7.0) |
| v4.0 | Sep 2026 | Dispersion calibration: VIX anchor, context block, shared Breadth weight, sign fixes (Report v8.0) |
