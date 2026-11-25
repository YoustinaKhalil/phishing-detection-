import base64

import streamlit as st

from feature_extraction import INVALID_URL_MESSAGE, is_valid_url, normalize_url
from predictor import phishing_probability, threshold

st.set_page_config(
    page_title="Link Checker",
    page_icon=":material/shield:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.html(
    """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {padding-top: 4rem; max-width: 720px;}
.hero {text-align: center; margin-bottom: 2rem;}
.hero .logo {line-height: 1; color: #60a5fa;}
.hero .logo img {width: 4rem; height: 4rem; display: inline-block;}
.hero h1 {margin: .6rem 0 .4rem 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -.5px;
  background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero p {margin: 0; opacity: .75; font-size: 1.1rem;}
div[data-testid="stTextInput"] input {border-radius: 14px; padding: .9rem 1.1rem; font-size: 1.05rem;}
div[data-testid="stFormSubmitButton"] button {border-radius: 14px; font-weight: 700; padding: .65rem 0; font-size: 1.05rem;}
.result {border-radius: 20px; padding: 2rem 1.8rem; text-align: center; border: 1px solid; margin-top: 1.5rem;}
.safe {background: rgba(34,197,94,.10); border-color: rgba(34,197,94,.55);}
.unsafe {background: rgba(239,68,68,.10); border-color: rgba(239,68,68,.55);}
.result .icon {line-height: 1;}
.result .icon img {width: 3.6rem; height: 3.6rem; display: inline-block;}
.safe .icon {color: #22c55e;}
.unsafe .icon {color: #ef4444;}
.result h2 {margin: .6rem 0 .3rem 0; font-size: 2rem; font-weight: 800;}
.result .desc {opacity: .85; font-size: 1.05rem;}
.result .url {margin-top: 1.1rem; font-family: ui-monospace, monospace; font-size: .85rem; opacity: .7;
  word-break: break-all; background: rgba(0,0,0,.25); padding: .55rem .8rem; border-radius: 10px;}
.note {text-align: center; opacity: .5; font-size: .85rem; margin-top: 1.6rem;}
</style>
"""
)


normalize = normalize_url


def is_phishing(u):
    return phishing_probability(u) >= threshold


# ───────────────────────── Icons (SVG as <img>, no emoji fonts needed) ─────────────────────────
def _icon(body: str, stroke: str) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="24" y2="24">'
        '<stop offset="0" stop-color="#60a5fa"/><stop offset="0.5" stop-color="#a78bfa"/>'
        '<stop offset="1" stop-color="#f472b6"/></linearGradient></defs>'
        f"{body}</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img alt="" src="data:image/svg+xml;base64,{b64}"/>'


SHIELD_SVG = _icon(
    '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1'
    'c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
    '<path d="m9 12 2 2 4-4"/>',
    "url(#g)",
)
ALERT_SVG = _icon(
    '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
    '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "#ef4444",
)
CHECK_SVG = _icon('<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>', "#22c55e")


# ───────────────────────── UI ─────────────────────────
st.html(
    f'<div class="hero"><div class="logo">{SHIELD_SVG}</div><h1>Is this link safe?</h1>'
    "<p>Paste any link below and we'll tell you if it's safe to open.</p></div>"
)

with st.form("check", border=False):
    url = st.text_input(
        "Link",
        placeholder="Paste a link here, e.g. https://example.com",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Check link", type="primary", width="stretch")

if submitted:
    if not url.strip():
        st.warning("Please paste a link first.")
    elif not is_valid_url(url):
        st.warning(INVALID_URL_MESSAGE)
    else:
        u = normalize(url)
        try:
            with st.spinner("Checking…"):
                bad = is_phishing(u)
        except Exception:
            st.error("This doesn't look like a valid link. Please check it and try again.")
        else:
            if bad:
                st.html(
                    f'<div class="result unsafe"><div class="icon">{ALERT_SVG}</div><h2>Not safe</h2>'
                    f'<div class="desc">This link looks like phishing. Don\'t open it or enter any passwords or personal info.</div>'
                    f'<div class="url">{u}</div></div>'
                )
            else:
                st.html(
                    f'<div class="result safe"><div class="icon">{CHECK_SVG}</div><h2>Safe</h2>'
                    f'<div class="desc">This link looks safe to open.</div>'
                    f'<div class="url">{u}</div></div>'
                )

st.html('<div class="note">No checker is 100% accurate — if something feels off, don\'t click.</div>')