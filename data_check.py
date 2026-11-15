"""
data_check.py — morning data check strip (MRS 8.3.0, 15 Nov 2026)
===================================================================
Everything on the page is fetched automatically (no manual inputs). This strip shows, at the top of the page, whether
each automatic source delivered the last completed NYSE session:
  teal  ✓  current  (data for the expected session)
  copper !  one session behind (a run is late or Yahoo was slow: reload in a few minutes)
  orange ✕  older or missing (a GitHub Actions run failed or a fetch keeps failing)
  grey  ·  waiting (a new log that has not had its first nightly run yet)
Expected session = the last NYSE session whose close has passed (New York time, 16:00), using the same holiday calendar
as the nightly update (rules copied from update_data.NYSECalendar). Pure functions; the app collects the dates.
"""
import html as _html
from datetime import datetime, timedelta

import pandas as pd

TEAL, COPPER, ORANGE, GREY, INK, MUTED = '#85cac5', '#e3b48e', '#f4a062', '#6e675f', '#f3ece2', '#a89f95'


from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, GoodFriday, USMartinLutherKingJr, USPresidentsDay,
                                    USMemorialDay, USLaborDay, USThanksgivingDay, nearest_workday, sunday_to_monday)


class _NYSE(AbstractHolidayCalendar):      # same rules as update_data.NYSECalendar (kept here so this module stands alone)
    rules = [Holiday('New Year', month=1, day=1, observance=sunday_to_monday), USMartinLutherKingJr, USPresidentsDay,
             GoodFriday, USMemorialDay, Holiday('Juneteenth', month=6, day=19, start_date='2022-01-01', observance=nearest_workday),
             Holiday('Independence Day', month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday('Christmas', month=12, day=25, observance=nearest_workday)]


_CAL = _NYSE()


def _trading(d: pd.Timestamp) -> bool:
    d = pd.Timestamp(d).normalize()
    return d.weekday() < 5 and len(_CAL.holidays(start=d, end=d)) == 0


def expected_session(now=None) -> pd.Timestamp:
    """The last NYSE session whose close (16:00 New York) has passed."""
    try:
        from zoneinfo import ZoneInfo
        now = now or datetime.now(ZoneInfo('America/New_York'))
    except Exception:
        now = now or datetime.utcnow() - timedelta(hours=5)
    d = pd.Timestamp(now.date())
    if now.hour < 16:
        d -= pd.Timedelta(days=1)
    while not _trading(d):
        d -= pd.Timedelta(days=1)
    return d.normalize()


def prev_session(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d) - pd.Timedelta(days=1)
    while not _trading(d):
        d -= pd.Timedelta(days=1)
    return d.normalize()


def status(date, expected: pd.Timestamp, pending: bool = False) -> str:
    if date is None or (isinstance(date, float) and date != date) or pd.isna(date):
        return 'waiting' if pending else 'missing'
    d = pd.Timestamp(date).normalize()
    if d >= expected:
        return 'ok'
    return 'late' if d >= prev_session(expected) else 'stale'


STYLE = {'ok': (TEAL, '✓'), 'late': (COPPER, '!'), 'stale': (ORANGE, '✕'), 'missing': (ORANGE, '✕'), 'waiting': (GREY, '·')}


def strip_html(items: list, issues: list, expected: pd.Timestamp) -> str:
    """items: [(name, date or None, tip, pending)], issues: [(section, why)] from the app's load-health list."""
    rows = [(n, d, tip, status(d, expected, pend)) for n, d, tip, pend in items]
    bad = [r for r in rows if r[3] in ('stale', 'missing')]
    late = [r for r in rows if r[3] == 'late']
    nb = len({r[0] for r in bad} | {a for a, _ in issues})
    if nb:
        head, hcol = f'{nb} ISSUE{"S" if nb != 1 else ""}', ORANGE
    elif late:
        head, hcol = f'{len(late)} LATE', COPPER
    else:
        head, hcol = 'ALL CURRENT', TEAL
    h = ('<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 8px 0;font-size:0.70rem;">'
         f'<span style="font-weight:800;letter-spacing:0.08em;color:#0a0a0a;background:{hcol};padding:2px 8px;'
         f'border-radius:3px;">DATA · {head}</span>'
         f'<span style="color:{MUTED};">expected session {expected:%a %d %b}</span>')
    for n, d, tip, st in rows:
        col, mark = STYLE[st]
        dt = '—' if st in ('missing', 'waiting') else f'{pd.Timestamp(d):%d %b}'
        t = _html.escape(f'{n}: {"no data" if st == "missing" else "first run pending" if st == "waiting" else dt}. {tip}', quote=True)
        h += (f'<span title="{t}" style="border:1px solid {col};color:{col};border-radius:3px;padding:1px 6px;'
              f'white-space:nowrap;">{mark} {_html.escape(n)} <span style="color:{MUTED};">{dt}</span></span>')
    h += '</div>'
    if issues:
        hl = ' · '.join(f'<b>{_html.escape(a)}</b> ({_html.escape(b)})' for a, b in issues)
        h += (f'<div style="font-size:0.74rem;color:{INK};border-left:3px solid {ORANGE};background:#2a1c13;'
              f'padding:4px 10px;margin:-4px 0 10px 0;border-radius:3px;">{hl}. <span style="color:{MUTED};">A reload in '
              f'a few minutes usually clears a fetch failure; a stale nightly file points to the GitHub Actions run.</span></div>')
    return h
