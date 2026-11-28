"""8.16.2: ETF ticker next to sector and industry names."""
import etf_names as en


def test_maps_and_labels():
    assert len(en.SECTOR_ETF) == 11 and len(en.INDUSTRY_ETF) >= 40
    assert en.plain('Financials') == 'Financials (XLF)' and en.plain('Regional Banks') == 'Regional Banks (KRE)'
    assert en.plain('Real Estate') == 'Real Estate (XLRE)' and en.plain('Comm Services') == 'Comm Services (XLC)'
    assert en.plain('Prepackaged Software') == 'Prepackaged Software'          # stock-level group: no ETF
    assert en.html('Banks') == 'Banks <span style="opacity:0.72;">(KBE)</span>'
    assert en.html('A&B') == 'A&amp;B'


def test_scan_row_and_heat_text():
    import scan_view, heat_score
    assert 'etf_names.html(sect)' in open(scan_view.__file__, encoding='utf-8').read()
    assert heat_score._etfn.plain('Energy') == 'Energy (XLE)'
