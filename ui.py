"""
ui.py — one component set for the whole MRS (8.4.0, 16 Nov 2026), ATN tokens (atn_brand.py)
=============================================================================================
Design audit 16 Nov 2026 (claude/mrs_design_8_4.md): every card now uses the same
  kicker   Barlow 500, 12.5px, 0.2em, uppercase, copper; the qualifier after "·" in grey
  tile     #161514, 1px #24211e, radius 6, padding 11 x 13; label 10.5px caps 0.14em grey
  value    IBM Plex Mono 600, 24px (hero 30px)
  tag      UPPERCASE 9.5px: TESTED copper fill · PROVISIONAL copper dashed · DESCRIPTIVE grey · MODELLED light copper
           dashed; any qualifier ("in sample", "v8 sizing") in grey text after it
  bar      10px track #1f1d1b, radius 3, white pin with a triangle head, mono tick labels
  colour   ATN tokens only (no reds): teal / teal muted = favourable, orange / orange muted = adverse,
           copper = brand and evidence, cream = neutral state
CSS() is injected once by app.py. It also restyles the older modules' classes (card .v8, Risk today .rp, allocation
.al, leaders .ld, Market detail .td / .ms) so they read as one system without rewriting their HTML.
"""
import html as _html

T = dict(ground='#0a0a0a', surface='#121212', tile='#161514', raised='#1a1918', line='#2a2724', line2='#3a3632',
         track='#1f1d1b', strong='#6e675f', ink='#f3ece2', muted='#a89f95', dim='#6e675f', copper='#e3b48e',
         copper_deep='#c58c69', copper_light='#f8d6b4', teal='#85cac5', teal_m='#649995', orange='#f4a062',
         orange_m='#b9805a', cream='#e8dccb', grey='#9b958f', neg1='#74492e', pos1='#3d5d5a')
DISPLAY = "Barlow, 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
_KINDS = {'tested': 'tested', 'provisional': 'prov', 'descriptive': 'desc', 'modelled': 'model', 'watch': 'model',
          'exploratory': 'desc', 'sized': 'tested', 'per': 'tested'}


def e(s):
    return _html.escape(str(s), quote=True)


def tag(text: str, kind: str = None) -> str:
    """Evidence tag. kind from the first word when not given ('Tested · in sample' -> TESTED + grey 'in sample')."""
    t = str(text).strip()
    first = t.replace('·', ' ').replace('(', ' ').split()[0].lower() if t else 'descriptive'
    k = _KINDS.get(kind or first, 'desc')
    head, rest = t, ''
    for sep in (' · ', ' (', ' + '):
        if sep in t:
            head, rest = t.split(sep, 1)[0], t[len(t.split(sep, 1)[0]):].strip(' ·(+').rstrip(')').strip()
            break
    if first in ('per', 'sized'):          # "Per trade · in sample" / "Sized · v8 run" -> TESTED + qualifier
        head, rest = 'Tested', t
    q = f'<span class="atn-q">{e(rest)}</span>' if rest else ''
    return f'<span class="atn-tagw"><span class="atn-tag atn-{k}">{e(head)}</span>{q}</span>'


def kicker(main: str, sub: str = '') -> str:
    return f'<div class="atn-k">{e(main)}' + (f' <b>· {e(sub)}</b>' if sub else '') + '</div>'


def bar(segs, pin=None, labels=(), height=10, margin=None) -> str:
    """segs: [(left %, width %, colour)], pin: % or None, labels: [(x %, text)]."""
    mg = f'margin:{margin};' if margin is not None else ''
    h = f'<div class="atn-bar" style="height:{height}px;{mg}">' + ''.join(
        f'<div class="atn-seg" style="left:{a:.2f}%;width:{w:.2f}%;background:{c};"></div>' for a, w, c in segs)
    if pin is not None:
        h += f'<div class="atn-pin" style="left:{min(max(pin, 0), 100):.2f}%"></div>'
    h += '</div>'
    if labels:
        h += '<div class="atn-lbls">' + ''.join(f'<span style="left:{x:.1f}%">{e(t)}</span>' for x, t in labels) + '</div>'
    return h


def CSS() -> str:
    t = T
    return f"""<style>
.atn-k{{font-family:{DISPLAY};font-weight:500;font-size:12.5px;line-height:16px;letter-spacing:.2em;text-transform:uppercase;color:{t['copper']};margin:0 0 10px 0}}
.atn-k b{{color:{t['muted']};font-weight:500}}
.atn-tile{{background:{t['tile']};border:1px solid #24211e;border-radius:6px;padding:11px 13px;min-width:0}}
.atn-tl{{display:flex;justify-content:space-between;align-items:center;gap:6px;font-family:{DISPLAY};font-weight:500;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:{t['muted']};margin-bottom:6px}}
.atn-val{{font-family:{MONO};font-weight:600;font-size:24px;line-height:1.15;color:{t['ink']}}}
.atn-vs{{font-family:{DISPLAY};font-size:12.5px;color:{t['muted']};font-weight:400}}
.atn-sm{{font-family:{DISPLAY};font-size:12.5px;color:{t['muted']};line-height:1.45}}
.atn-tagw{{display:inline-flex;align-items:center;gap:5px;white-space:nowrap;float:none}}
.atn-tag{{font-family:{DISPLAY};font-weight:600;font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;padding:0 6px;border-radius:3px;border:1px solid;line-height:15px}}
.atn-tested{{color:{t['copper']};border-color:{t['copper']};background:rgba(227,180,142,.13)}}
.atn-prov{{color:{t['copper']};border-color:{t['copper']};border-style:dashed;background:transparent}}
.atn-desc{{color:{t['muted']};border-color:#4a4540;background:transparent}}
.atn-model{{color:{t['copper_light']};border-color:{t['copper_light']};border-style:dashed;background:transparent}}
.atn-q{{font-family:{DISPLAY};font-size:10px;color:{t['dim']};letter-spacing:.04em;text-transform:none}}
.atn-bar{{position:relative;background:{t['track']};border-radius:3px;margin:9px 0 3px 0}}
.atn-seg{{position:absolute;top:0;bottom:0}}
.atn-bar .atn-seg:first-child{{border-radius:3px 0 0 3px}}
.atn-pin{{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1px;background:{t['ink']};border-radius:1px}}
.atn-pin:before{{content:'';position:absolute;top:-6px;left:-4px;border-left:5.5px solid transparent;border-right:5.5px solid transparent;border-top:6px solid {t['ink']}}}
.atn-lbls{{position:relative;height:13px;font-family:{MONO};font-size:10px;color:{t['dim']}}}
.atn-lbls span{{position:absolute;transform:translateX(-50%);white-space:nowrap}}
.atn-chip{{display:inline-block;font-family:{DISPLAY};font-weight:600;font-size:12px;padding:3px 9px;border-radius:4px;background:#24211e;color:{t['ink']};white-space:nowrap;margin:2px 3px 2px 0}}
.atn-ring{{width:96px;height:96px;border-radius:50%;display:grid;place-items:center}}
.atn-ring.sm{{width:78px;height:78px}}
.atn-ring.sm > div{{width:60px;height:60px;font-size:16px}}
.atn-ring > div{{width:74px;height:74px;border-radius:50%;background:{t['tile']};display:grid;place-items:center;font-family:{MONO};font-weight:600;font-size:18px;color:{t['ink']}}}
.atn-row .pill{{display:inline-block;min-width:40px;text-align:center;font-family:{DISPLAY};font-size:12px;font-weight:700;padding:3px 7px;margin:2px 4px 2px 0;border-radius:4px;border:1px solid {t['line2']};color:{t['dim']}}}
.atn-row .dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
.atn-row .sm{{font-family:{DISPLAY};font-size:12.5px;color:{t['muted']};line-height:1.5}}
.atn-row .ft{{font-size:11px;color:{t['dim']};margin-top:4px}}
@media (max-width:900px){{.atn-row{{grid-template-columns:1fr !important}}}}
/* bordered Streamlit containers = card shell; nested ones = tiles */
[data-testid="stVerticalBlockBorderWrapper"]{{border-color:{t['line']} !important;border-radius:6px !important;background:{t['surface']}}}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"]{{background:{t['tile']};border-color:#24211e !important}}
/* ---- harmonise the older modules (no coloured left borders; one tag; one bar; mono numbers) ---- */
.v8,.rp,.al,.td,.ms,.ds{{border-left:1px solid {t['line']} !important}}
.v8 .k,.rp .lab,.ld .hdr .ttl{{font-family:{DISPLAY} !important;font-weight:500 !important;font-size:10.5px !important;letter-spacing:.14em !important;text-transform:uppercase !important}}
.rp > .k,.al .k,.td .k{{font-family:{DISPLAY} !important;font-weight:500 !important;font-size:12.5px !important;letter-spacing:.2em !important;color:{t['copper']} !important}}
.ld .hdr .ttl{{font-size:12.5px !important;letter-spacing:.2em !important}}
.v8 .val,.rp .big{{font-family:{MONO} !important;font-weight:600 !important;font-size:24px !important}}
.v8 .bar,.rp .trk{{height:10px !important;border-radius:3px !important}}
.v8 .pin{{top:-4px !important;height:18px !important}}
.v8 .pin:before{{content:'';position:absolute;top:-6px;left:-4px;border-left:5.5px solid transparent;border-right:5.5px solid transparent;border-top:6px solid {t['ink']}}}
.v8 .tile,.rp .tile{{background:{t['tile']} !important;border:1px solid #24211e !important;border-radius:6px !important}}
</style>"""
