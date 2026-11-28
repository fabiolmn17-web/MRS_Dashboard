"""8.16.2 (28 Nov 2026): the ETF ticker next to every sector and industry name on the page.

Sector and industry names come from StockScanner/industry.py (INDUSTRY_MAP, SECTOR_ETF), the same table the nightly
industry scan uses, so the names here always match the ones in the data. Stock-level industry groups (scanner) have no
ETF and are shown without a ticker.
"""
import html as _h

try:
    from StockScanner.industry import INDUSTRY_MAP as _IM, SECTOR_ETF as _SE
except Exception:                                    # the app must render even if the scanner package is missing
    _IM, _SE = {}, {}

SECTOR_ETF = dict(_SE)
INDUSTRY_ETF = {name: etf for rows in _IM.values() for etf, name in rows}
_ALL = {**INDUSTRY_ETF, **SECTOR_ETF}


def etf(name) -> str:
    """The ETF ticker for a sector or industry display name, or '' when there is none."""
    return _ALL.get(str(name).strip(), '')


def plain(name) -> str:
    """'Financials (XLF)'; the name alone when there is no ETF."""
    t = etf(name)
    return f'{name} ({t})' if t else str(name)


def html(name, escape: bool = True) -> str:
    """HTML: the name, then the ticker in brackets, slightly dimmed (inherits the colour around it)."""
    t = etf(name)
    n = _h.escape(str(name)) if escape else str(name)
    return f'{n} <span style="opacity:0.72;">({t})</span>' if t else n
