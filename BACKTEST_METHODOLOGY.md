# MRS Backtest Framework — Methodology

**Author:** Epistruct Research  
**Date:** July 2026  
**Version:** 2.0

---

## Overview

This backtest evaluates how the Market Regime Score (MRS) algorithm behaves around significant market drawdowns. It analyzes both **sell-side signals** (early warning before peaks) and **buy-side signals** (recovery detection at bottoms).

---

## Files

| File | Description |
|------|-------------|
| `backtest_mrs.py` | Main backtest script — computes MRS, finds drawdowns, analyzes events |
| `backtest_report.html` | Interactive HTML report with findings and visualizations |
| `Data/AMEX_SPY, 1D_066f4.csv` | Source data (SPY, VIX, SKEW, B20%, PC Ratio, ADL, Volume) |

---

## Data Requirements

The backtest uses daily data with the following columns:

| Column | Description | Source |
|--------|-------------|--------|
| `close` | SPY daily close | TradingView |
| `vix` | VIX index close | TradingView (TVC) |
| `skew` | CBOE SKEW index | TradingView (CBOE) |
| `b20_pct` | % stocks above 20 SMA (S5TW) | TradingView (INDEX) |
| `pc_ratio` | Put/Call ratio | TradingView (USI) |
| `adl` | Advance-Decline Line | TradingView |
| `volume` | SPY daily volume | TradingView |

**Data availability:** Full analysis runs from **December 2006** onward (when B20% and PC Ratio become available, plus 756-day Phi warm-up period).

---

## Methodology

### 1. MRS Computation

The MRS is computed using the same logic as `pipeline.py`:

1. **Phi Calculation**: 756-day (3-year) rolling percentile rank for each indicator
2. **Component Scoring**: Each component scored based on Phi thresholds
3. **Composite Score**: Sum of all component scores (range: approx -6.5 to +4.0)

**Components:**
- VIX (volatility regime)
- Extension (price vs SMA50)
- Momentum (20-day price change)
- ADL (breadth, 20-day ROC)
- B20% (% stocks above 20 SMA)
- PC Ratio (5-zone model)
- SKEW (tail risk)
- Zero Gamma (excluded — no historical data)

**Regime Labels:**

| MRS Score | Regime Label |
|-----------|--------------|
| ≥ 1.5 | RISK-ON |
| 0.5 to 1.5 | MILD RISK-ON |
| -0.5 to 0.5 | NEUTRAL |
| -1.5 to -0.5 | MILD RISK-OFF |
| < -1.5 | RISK-OFF |

### 2. Drawdown Detection

**Definition:**
- **Peak**: Local high where price doesn't exceed for at least 20 trading days
- **Trough**: Lowest point before price recovers 50% of the drawdown
- **Valid Event**: Peak-to-trough decline within severity band

**Overlapping Events**: When drawdowns overlap, the larger one is kept.

### 3. Mutually Exclusive Severity Bands (v2.0 Change)

**Previous approach (v1.0)**: Nested thresholds (≥5%, ≥10%, ≥15%, ≥20%) where a 52% crash counted in ALL four buckets.

**Problem**: This inflated counts at lower thresholds and made interpretation difficult. A routine 6% pullback was mixed with catastrophic bear markets in the "≥5%" analysis.

**New approach (v2.0)**: Mutually exclusive bands — each drawdown appears in exactly ONE category:

| Band | Range | Description | Typical Events |
|------|-------|-------------|----------------|
| 5-10% | 5% ≤ DD < 10% | Minor Correction | Routine pullbacks |
| 10-15% | 10% ≤ DD < 15% | Correction | Standard corrections |
| 15-20% | 15% ≤ DD < 20% | Significant Correction | Borderline bear territory |
| 20%+ | DD ≥ 20% | Bear Market | Major drawdowns (2008, 2022) |

**Benefit**: Component behavior at each severity level is now "pure" — you can see if MRS behaves differently for minor pullbacks vs bear markets.

### 4. Analysis Windows

| Window | Description |
|--------|-------------|
| **Pre-Peak 30d** | 30 trading days before peak |
| **Pre-Peak 60d** | 60 trading days before peak |
| **Pre-Peak 90d** | 90 trading days before peak |
| **During Drawdown** | Peak to trough |
| **Recovery** | Trough to 50% retracement (or 60 days) |

### 5. Metrics Computed

#### Sell-Side (Early Warning)
- MRS at peak (score and regime)
- Days MRS was negative before peak (lead time)
- Days MRS was in RISK-OFF before peak
- MRS trajectory (slope) pre-peak
- Component deterioration sequence
- Volume divergence (price up, volume down)

**Lead Time Definition**: Number of days before the peak that MRS first went negative (within the analysis window). Example: lead time of 54 days means MRS turned negative 54 days before the market peaked.

#### During Drawdown
- Minimum MRS reached
- Average MRS
- % days in RISK-OFF
- Regime transitions count
- Maximum VIX

#### Buy-Side (Recovery)
- Days for MRS to turn positive after trough
- Days to exit RISK-OFF (MRS > -0.5)
- Component recovery sequence (which turns positive first)
- Price recovery milestones (25%, 50%, 75%, 100%)
- Volume confirmation (higher volume on up days)
- Capitulation detection (volume spike at trough)

### 6. Volume Analysis

#### Raw Volume Divergence
**Definition**: Price rising (60-day change > 0) while volume declining (60-day change < -10%).

**Pre-Peak:**
- Volume trend vs price trend (divergence detection)
- Price-volume correlation
- Volume percentile at peak

**At Trough:**
- Volume ratio vs 20-day and 50-day averages
- Capitulation flag (volume > 2x average)

**During Recovery:**
- Volume on up days vs down days
- Volume trend slope

#### Seasonal Volume Normalization (Tested and Rejected)

**Hypothesis**: Summer (July-August) and holiday periods have structurally lower volume. A "volume divergence" during these periods might be a false signal — just vacation effect, not distribution.

**Implementation**: Day-of-year percentile comparison:
- Compares today's volume to historical volume for the same calendar week (±1 week window)
- Uses 5 years of lookback history
- Seasonal divergence flagged when: price up AND volume < 40th percentile for this season

**Statistical Testing**: See Section 7 below.

**Result**: Seasonal normalization **removed important information** and performed worse than raw divergence. Raw volume divergence should be used.

### 7. Volume Signal Statistical Testing

We tested whether raw or seasonally-normalized volume divergence better predicts drawdowns.

#### Tests Performed

**1. Chi-Square Test of Independence**

Tests whether there is a statistically significant association between the signal firing and a drawdown occurring within 60 days.

- **Null hypothesis (H₀)**: Signal and drawdown occurrence are independent
- **Alternative hypothesis (H₁)**: Signal and drawdown occurrence are associated
- **Significance level**: α = 0.05

**Why Chi-Square?** Appropriate for categorical data (signal fires: yes/no vs drawdown follows: yes/no). Tests whether the observed frequencies differ significantly from what we'd expect if the variables were independent.

**2. Contingency Table Analysis**

For each signal type, we computed:

|  | Drawdown Follows | No Drawdown |
|--|------------------|-------------|
| Signal fires | True Positive (TP) | False Positive (FP) |
| No signal | False Negative (FN) | True Negative (TN) |

**Metrics derived:**
- **Precision** = TP / (TP + FP) — When signal fires, how often is it right?
- **Recall** = TP / (TP + FN) — Of all drawdowns, how many did signal catch?
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) — Balanced measure
- **Odds Ratio** = (TP × TN) / (FP × FN) — How much more likely is drawdown when signal fires?
- **Lift** = Precision / Base Rate — How much better than random guessing?

**3. Forward Return Analysis**

Measured actual 60-day forward returns after each signal type to validate predictive value.

#### Results

| Metric | Raw Divergence | Seasonal Divergence | Winner |
|--------|----------------|---------------------|--------|
| Precision | 50.7% | 39.8% | **Raw** |
| Recall | 39.1% | 32.5% | **Raw** |
| F1 Score | 44.2 | 35.8 | **Raw** |
| Odds Ratio | **1.24** | 0.61 | **Raw** |
| Chi-sq p-value | 4.1e-04 | 6.9e-17 | Both significant |

**Key Finding**: Seasonal divergence has odds ratio < 1, meaning it's actually a **contrarian indicator** — low seasonal volume is associated with BETTER forward returns:

| Seasonal Volume Regime | Avg 60-day Forward Return | % Positive |
|------------------------|---------------------------|------------|
| Low (<40th percentile) | **+2.66%** | **75.8%** |
| High (>60th percentile) | +1.15% | 60.8% |

**Interpretation**: Low seasonal volume indicates calm, low-participation markets — which tend to have better returns and less volatility. The "noise" we thought we were removing is actually information.

**At Actual Drawdown Peaks:**
- Raw divergence firing: **26 of 45 (58%)**
- Seasonal divergence firing: 15 of 45 (33%)

#### Conclusion

**Use raw volume divergence, not seasonal.** Seasonal normalization removes important information and degrades predictive power.

### 8. False Positive Analysis

Tracks periods where MRS < 0 but no significant drawdown followed within 60 days.

**Metrics:**
- Total negative MRS periods
- True positives (preceded drawdown)
- False positives (no drawdown followed)
- Precision = TP / (TP + FP)

---

## Key Findings (July 2026 Run)

### Drawdown Distribution by Band

| Band | Description | Events |
|------|-------------|--------|
| 5-10% | Minor Correction | 37 |
| 10-15% | Correction | 3 |
| 15-20% | Significant Correction | 3 |
| 20%+ | Bear Market | 2 |
| **Total** | | **45** |

### MRS Warning Capability by Band

| Band | Events | MRS < 0 at Peak | RISK-OFF at Peak | Avg Lead Time |
|------|--------|-----------------|------------------|---------------|
| 5-10% | 37 | 22% (8) | 8% (3) | 54 days |
| 10-15% | 3 | 33% (1) | 33% (1) | 60 days |
| 15-20% | 3 | 33% (1) | 0% (0) | 43 days |
| 20%+ | 2 | 0% (0) | 0% (0) | 58 days |

### Granular Regime Breakdown at Peak (5-10% Band)

| Regime | Count | % | MRS Range |
|--------|-------|---|-----------|
| RISK-ON | 18 | 48.6% | +1.5 to +4.0 |
| MILD RISK-ON | 5 | 13.5% | +0.5 to +1.0 |
| NEUTRAL | 11 | 29.7% | -0.5 to 0.0 |
| MILD RISK-OFF | 0 | 0% | — |
| RISK-OFF | 3 | 8.1% | -2.5 to -5.5 |

The "22% MRS negative" = 8 events: 5 in lower NEUTRAL (-0.5 to 0) + 3 in RISK-OFF.

### Bear Market Finding

Both major bear markets (2008 GFC -52%, 2022 -21%) started when MRS was **MILD RISK-ON (+0.5)**. MRS did not warn at the peak, but the lead time metric shows it went negative during the decline (~58 days before trough on average).

**Interpretation**: MRS is better at **confirming drawdowns in progress** than predicting exact tops.

### Volume Divergence

Raw volume divergence (price up, volume down >10%) was present at **58% of drawdown peaks** (26 of 45 events).

---

## Configuration

Edit constants at top of `backtest_mrs.py`:

```python
# Mutually exclusive drawdown severity bands
DRAWDOWN_BANDS = [
    (0.05, 0.10, "5-10%", "Minor Correction"),
    (0.10, 0.15, "10-15%", "Correction"),
    (0.15, 0.20, "15-20%", "Significant Correction"),
    (0.20, 1.00, "20%+", "Bear Market"),
]

PRE_PEAK_WINDOWS = [30, 60, 90]   # Analysis windows (days)
VOLUME_WINDOWS = [20, 50]         # Volume normalization windows
PHI_WINDOW = 756                  # Rolling percentile window (3 years)
SEASONAL_VOLUME_YEARS = 5         # Years for seasonal baseline (tested, not used)
```

---

## Running the Backtest

```bash
cd MRS_WebApp
python backtest_mrs.py
```

Output: `backtest_report.html` (interactive report)

---

### 9. Component Weight Calibration (MRS v2.0)

Weights derived from component timing analysis during drawdown events:

**Methodology:**
1. For each component, measured "lead time" = days MRS went negative before peak (when that component was the driver)
2. Measured "recovery time" = days for component to turn positive after trough
3. Components with better sell-side timing get higher weights for their negative scores
4. Components with better buy-side timing get higher weights for their positive scores

**Final Weights (v2.0):**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| VIX | 1.3 | Strong sell-side (2nd best lead time, most first-to-warn) |
| Extension | 1.2 | Good sell-side (3rd best lead time) |
| Momentum | 1.0 | Average timing |
| ADL | 1.0 | Average timing |
| B20% | 1.1 | Good buy-side (2nd fastest recovery) |
| PC Ratio | 1.4 | Best buy-side (fastest recovery, most first-to-recover) |
| SKEW | 1.3 | Best sell-side lead time |
| Gamma | 1.0 | No timing data available |
| Volume | 1.0 | Volume divergence (sell-side only, -0.5 when divergent) |

**Volume Divergence Component:**
- Condition: Price up (60d) AND volume down >10% (60d)
- Score: -0.5 when divergent, 0.0 otherwise
- Statistical validation: Present at 58% of drawdown peaks

### 10. Robustness Testing

Tested MRS v2.0 improvements across multiple Phi windows (1yr, 2yr, 3yr, 5yr) to ensure the calibration is not overfit to the 3-year default.

**Definition:**
- Phi Window = rolling lookback period for percentile rank calculation
- 1yr = 252 trading days, 2yr = 504, 3yr = 756 (default), 5yr = 1260

**Results:**

| Phi Window | Warning Rate | Precision | True Positives | Improvement vs v1 |
|------------|--------------|-----------|----------------|-------------------|
| 1yr (252d) | 48.9% | 48.9% | 142 | +22.2% / +1.1% |
| 2yr (504d) | 44.4% | 50.9% | 140 | +20.0% / +2.3% |
| **3yr (756d)** | **40.0%** | **52.5%** | **149** | **+17.8% / +2.5%** |
| 5yr (1260d) | 35.6% | 51.4% | 143 | +11.1% / +1.6% |

**Key Finding:** Improvements hold across ALL Phi windows. The v2.0 calibration is robust, not overfit.

**Optimal Window:** 3-year (756d) remains the best balance:
- Highest true positives (149)
- Best precision (52.5%)
- Reasonable warning rate (40%)

Shorter windows (1yr, 2yr) generate more signals but with lower precision. Longer windows (5yr) are too conservative.

---

## MRS v2.0 Final Specification

**Implemented in `pipeline.py` (July 19, 2026)**

```python
# Component Weights
COMPONENT_WEIGHTS = {
    'vix':  1.3,   # Strong sell-side
    'ext':  1.2,   # Good sell-side
    'mom':  1.0,   # Average timing
    'adl':  1.0,   # Average timing
    'b20':  1.1,   # Good buy-side
    'pc':   1.4,   # Best buy-side
    'skew': 1.3,   # Best sell-side
    'gamma': 1.0,  # No timing data
    'vol':  1.0,   # Volume divergence
}

# Volume Divergence
def score_volume_divergence(price_60d_chg, vol_60d_chg):
    if price_60d_chg > 0 and vol_60d_chg < -0.10:
        return -0.5, 'Divergence (bearish)'
    return 0.0, 'Normal'
```

**Performance (3yr Phi window):**
- Warning Rate: 40.0% (vs 22.2% v1) — +17.8%
- Precision: 52.5% (vs 50.2% v1) — +2.3%
- True Positives: 149 (vs 129 v1) — +20 events

---

## Next Iteration

- [x] ~~Integrate raw volume divergence into MRS scoring~~ ✓ Done
- [x] ~~Test component weight adjustments~~ ✓ Done
- [x] ~~Validate robustness across Phi windows~~ ✓ Done
- [x] ~~Add recovery signal to dashboard~~ ✓ Done
- [ ] Backtest with Zero Gamma (if historical data becomes available)
- [ ] Analyze sector rotation patterns around drawdowns

---

## Changelog

### v2.1 (July 19, 2026)
- Added component weights based on timing analysis
- Integrated volume divergence as new MRS component (-0.5 when bearish)
- Validated robustness across 1yr, 2yr, 3yr, 5yr Phi windows
- Updated `pipeline.py` with MRS v2.0 specification
- Confirmed 3-year Phi window as optimal
- Added recovery signal to dashboard (`compute_recovery_signal()` in pipeline.py)
  - Monitors PC Ratio (fastest recovery), B20%, VIX spike zone
  - Fires when MRS negative + 2+ recovery components active
  - Strength levels: STRONG (4+), MODERATE (2-3), WEAK (1)

### v2.0 (July 19, 2026)
- Changed from nested thresholds to mutually exclusive severity bands
- Implemented seasonal volume normalization
- Conducted statistical testing (Chi-square, contingency tables, forward returns)
- **Rejected** seasonal normalization based on statistical evidence
- Added granular regime breakdown at peak
- Documented statistical methodology

### v1.0 (July 2026)
- Initial implementation with nested thresholds
