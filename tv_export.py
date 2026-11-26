"""TradingView watchlist export (8.13.0, 26 Nov 2026).

TradingView imports a .txt watchlist (watchlist menu → Import list): symbols separated by commas, and an optional
section header written as ###NAME. Class shares use a dot on TradingView (BRK.B), a dash in our data (BRK-B).
"""
import re
import pandas as pd


def tv_symbol(t) -> str:
    """Our ticker → TradingView symbol (BRK-B → BRK.B). Empty for missing values."""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ''
    s = str(t).strip().upper()
    if s in ('', 'NAN', 'NONE'):
        return ''
    return s.replace('-', '.')


def section_name(name: str) -> str:
    """A TradingView section name: no commas or #, single spaces, upper case, at most 60 characters."""
    s = re.sub(r'[,#\n\r]+', ' ', str(name or '')).replace('·', '-').strip()
    s = s.encode('ascii', 'ignore').decode()                # plain ASCII for the TradingView importer
    s = re.sub(r'\s+', ' ', s).upper()
    return s[:60]


def tv_text(tickers, section: str = None) -> str:
    """Import text for one list, in the given order, duplicates dropped. '' when the list is empty."""
    seen, out = set(), []
    for t in tickers if tickers is not None else []:
        s = tv_symbol(t)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if not out:
        return ''
    head = section_name(section) if section else ''
    return ','.join(([f'###{head}'] if head else []) + out)


def file_name(label: str, day: str) -> str:
    """File name for the download: MRS_<label>_<day>.txt with a safe label."""
    s = re.sub(r'[^A-Za-z0-9]+', '_', str(label or 'list')).strip('_')
    return f'MRS_{s or "list"}_{day}.txt'
