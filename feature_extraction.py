"""URL feature extraction for the phishing detector.

* Works fully offline (no tldextract / no downloads needed).
* The SAME code is used for building the dataset, training, the API and the app,
  so there is no train/serve mismatch.
"""
import ipaddress
import math
import os
import re
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# TLD / public-suffix handling (offline)
# ---------------------------------------------------------------------------
_CCTLD = set(
    "ac ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi bj bm bn bo br bs bt bw by bz "
    "ca cc cd cf cg ch ci ck cl cm cn co cr cu cv cw cx cy cz de dj dk dm do dz ec ee eg er es et eu fi fj fk fm "
    "fo fr ga gd ge gf gg gh gi gl gm gn gp gq gr gs gt gu gw gy hk hm hn hr ht hu id ie il im in io iq ir is it "
    "je jm jo jp ke kg kh ki km kn kp kr kw ky kz la lb lc li lk lr ls lt lu lv ly ma mc md me mg mh mk ml mm mn "
    "mo mp mq mr ms mt mu mv mw mx my mz na nc ne nf ng ni nl no np nr nu nz om pa pe pf pg ph pk pl pm pn pr ps "
    "pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sk sl sm sn so sr ss st su sv sx sy sz tc td tf tg th tj "
    "tk tl tm tn to tr tt tv tw tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws ye yt za zm zw".split()
)
_SLD = set(
    "co com org net gov edu ac or ne go gob gv mil nhs sch ltd plc id info biz web nom art "
    "gouv govt police mod parliament school".split()
)


def _load_tlds():
    tlds = set(_CCTLD)
    path = os.path.join(HERE, "data", "known_tlds.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tlds |= {x.strip().lower() for x in f if x.strip()}
    return tlds


KNOWN_TLDS = _load_tlds()


def split_host(host: str):
    """host -> (subdomain, domain_label, suffix).  Offline public-suffix heuristic."""
    labels = [x for x in host.lower().strip(".").split(".") if x]
    if len(labels) == 1:
        return "", labels[0], ""
    suffix_n = 1
    if len(labels) >= 3 and labels[-1] in _CCTLD and labels[-2] in _SLD:
        suffix_n = 2
    suffix = ".".join(labels[-suffix_n:])
    rest = labels[:-suffix_n]
    if not rest:
        return "", "", suffix
    return ".".join(rest[:-1]), rest[-1], suffix


def ent(s):
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


# ---------------------------------------------------------------------------
# Knowledge lists
# ---------------------------------------------------------------------------
KW = [
    "login", "signin", "secure", "account", "update", "verify", "bank", "paypal",
    "confirm", "free", "bonus", "webscr", "support", "wallet", "password",
    "service", "ebay", "apple", "microsoft",
]
KW_EXTRA = [
    "auth", "billing", "invoice", "payment", "suspend", "unlock", "recover", "restore",
    "claim", "reward", "prize", "gift", "airdrop", "crypto", "validate", "alert",
    "security", "customer", "helpdesk", "limited", "official", "online", "mobile",
    "signon", "logon", "credential", "otp", "token", "kyc", "refund", "delivery", "tracking",
]

# brand name -> its real registered domains
BRANDS = {
    "paypal": {"paypal.com"}, "apple": {"apple.com", "icloud.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com", "outlook.com"},
    "google": {"google.com", "gmail.com", "googleapis.com"}, "amazon": {"amazon.com", "amazon.co.uk", "amazon.de", "aws.amazon.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"}, "instagram": {"instagram.com"},
    "netflix": {"netflix.com"}, "ebay": {"ebay.com", "ebay.co.uk"}, "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"}, "whatsapp": {"whatsapp.com"}, "twitter": {"twitter.com", "x.com"},
    "chase": {"chase.com"}, "wellsfargo": {"wellsfargo.com"}, "bankofamerica": {"bankofamerica.com"},
    "citibank": {"citibank.com", "citi.com"}, "hsbc": {"hsbc.com", "hsbc.co.uk"},
    "dhl": {"dhl.com"}, "fedex": {"fedex.com"}, "ups": {"ups.com"}, "usps": {"usps.com"},
    "binance": {"binance.com"}, "coinbase": {"coinbase.com"}, "metamask": {"metamask.io"},
    "ledger": {"ledger.com"}, "trustwallet": {"trustwallet.com"}, "steam": {"steampowered.com", "steamcommunity.com"},
    "adobe": {"adobe.com"}, "docusign": {"docusign.com", "docusign.net"}, "zoom": {"zoom.us"},
    "spotify": {"spotify.com"}, "telegram": {"telegram.org", "t.me"}, "outlook": {"outlook.com", "office.com", "live.com"},
    "office365": {"office.com", "microsoft.com"}, "yahoo": {"yahoo.com"}, "alibaba": {"alibaba.com"},
    "walmart": {"walmart.com"}, "americanexpress": {"americanexpress.com"}, "visa": {"visa.com"},
    "mastercard": {"mastercard.com"}, "santander": {"santander.com"}, "barclays": {"barclays.co.uk", "barclays.com"},
    "vodafone": {"vodafone.com", "vodafone.co.uk"}, "orange": {"orange.fr", "orange.com"},
    "github": {"github.com"}, "icloud": {"icloud.com", "apple.com"}, "roblox": {"roblox.com"},
    "discord": {"discord.com", "discord.gg"}, "tiktok": {"tiktok.com"},
}
BRAND_NAMES = sorted(BRANDS)

FREE_HOSTS = {
    "pages.dev", "weebly.com", "github.io", "web.app", "firebaseapp.com", "wixsite.com", "blogspot.com",
    "glitch.me", "netlify.app", "000webhostapp.com", "duckdns.org", "workers.dev", "vercel.app",
    "herokuapp.com", "ngrok.io", "r2.dev", "webflow.io", "godaddysites.com", "sites.google.com",
    "myshopify.com", "wordpress.com", "square.site", "carrd.co", "typedream.app", "framer.app",
    "repl.co", "replit.app", "onrender.com", "surge.sh", "fleek.co", "ipfs.io", "azurewebsites.net",
    "cloudfront.net", "appspot.com", "bubbleapps.io", "wix.com", "jimdofree.com", "webcindario.com",
    "hsforms.com", "jotform.com", "forms.gle", "ead.me", "campaign-archive.com", "mailchimpsites.com",
    "strikingly.com", "yolasite.com", "my.canva.site", "notion.site", "gitbook.io", "tiiny.site",
    "ddns.net", "no-ip.org", "hopto.org", "zapto.org", "serveo.net", "trycloudflare.com",
}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy",
    "qrco.de", "q-r.to", "shorturl.at", "tiny.cc", "rebrand.ly", "lnkd.in", "s.id", "v.gd",
    "t.ly", "bl.ink", "short.io", "adf.ly", "bitly.com", "shorte.st", "clck.ru", "ln.run", "urlz.fr",
}
SUSP_TLDS = {
    "xyz", "top", "icu", "click", "buzz", "cfd", "sbs", "live", "cc", "tk", "ml", "ga", "cf", "gq",
    "work", "support", "rest", "monster", "lol", "cyou", "bond", "quest", "vip", "shop", "site",
    "online", "store", "link", "win", "bid", "loan", "date", "men", "party", "review", "stream",
    "download", "racing", "accountant", "faith", "science", "country", "kim", "zip", "mov", "wang",
    "pw", "ws", "su", "ink", "fun", "life", "world", "today", "space", "website", "info", "biz",
}
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "co.uk", "de", "fr", "jp", "io", "ac.uk", "nhs.uk"}
SENSITIVE_EXT = re.compile(r"\.(exe|scr|zip|rar|apk|msi|bat|cmd|jar|js|vbs|iso|dmg|pif|docm|xlsm)(\?|$|/)", re.I)

_HOMOGLYPH = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "$": "s"})


def _deleet(s: str) -> str:
    s = s.lower().translate(_HOMOGLYPH)
    return s.replace("rn", "m").replace("vv", "w")


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Add a scheme if the user omitted it."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


_LABEL_RE = re.compile(r"^[^\W_](?:[\w-]{0,61}[^\W_])?$", re.UNICODE)
_NON_WEB_SCHEMES = re.compile(r"^(mailto|tel|sms|javascript|data|file|ftp):", re.I)

INVALID_URL_MESSAGE = (
    "This doesn't look like a link. "
    "Please paste a full web address, e.g. https://example.com"
)


def _is_numeric_ip(host: str) -> bool:
    """decimal (3232235777) / hex (0xC0A80001) forms of an IP address"""
    return bool(re.fullmatch(r"\d{8,10}", host) or re.fullmatch(r"0x[0-9a-f]{6,8}", host, re.I)
                or (re.fullmatch(r"(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){3}", host, re.I) and "0x" in host.lower()))


def is_valid_url(raw: str) -> bool:
    """True only if `raw` looks like a real web link (http/https + domain with a
    known TLD, or an IP address)."""
    raw = (raw or "").strip()
    if not raw or re.search(r"\s", raw):
        return False
    if _NON_WEB_SCHEMES.match(raw):
        return False
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", raw)
    if m and m.group(1).lower() not in ("http", "https"):
        return False
    if not m and "@" in raw.split("/")[0]:
        return False
    try:
        parts = urlsplit(normalize_url(raw))
        host = parts.hostname
        parts.port
    except ValueError:
        return False
    if not host:
        return False
    host = host.rstrip(".")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if _is_numeric_ip(host):
        return True
    labels = host.split(".")
    if len(labels) < 2:
        return False
    tld = labels[-1].lower()
    if not (tld in KNOWN_TLDS or tld.startswith("xn--")):
        return False
    return all(_LABEL_RE.match(label) for label in labels)


# ---------------------------------------------------------------------------
# Trusted domains (only the big, stable ones - NOT user-content hosts)
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "wikipedia.org", "amazon.com", "microsoft.com", "github.com",
    "linkedin.com", "apple.com", "mozilla.org", "stackoverflow.com", "python.org", "wikimedia.org",
    "cloudflare.com", "oracle.com", "ibm.com", "adobe.com", "netflix.com", "paypal.com",
}
# Sub-domains of trusted domains where ANYONE can publish content -> never auto-trusted
USER_CONTENT_SUBDOMAINS = {
    "docs", "drive", "sites", "forms", "storage", "script", "firebase", "groups", "sheets", "slides",
    "pages", "gist", "raw", "user-images", "blogger", "photos", "translate", "webcache", "cache",
    "play", "app", "apps", "calendar", "mail",
}


def registered_domain(url: str) -> str:
    """google.com.attacker.net -> attacker.net  (so tricks don't pass)"""
    host = (urlsplit(normalize_url(url)).hostname or "").rstrip(".")
    sub, dom, suf = split_host(host)
    return f"{dom}.{suf}" if suf and dom else (dom or suf)


def is_trusted(url: str) -> bool:
    parts = urlsplit(normalize_url(url))
    host = (parts.hostname or "").rstrip(".")
    if "@" in parts.netloc:
        return False
    sub, dom, suf = split_host(host)
    if f"{dom}.{suf}" not in TRUSTED_DOMAINS:
        return False
    return not any(l in USER_CONTENT_SUBDOMAINS for l in (sub.split(".") if sub else []))


def inner_url(url: str):
    """If the query holds another link (redirect), return it, else None."""
    for _, v in parse_qsl(urlsplit(normalize_url(url)).query):
        if re.match(r"^https?://", v, re.I) and is_valid_url(v):
            return v
    return None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def extract_features_dict(url: str) -> dict:
    url = normalize_url(url)
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    host = (parts.hostname or "").rstrip(".").lower()
    netloc = parts.netloc
    path, query, fragment = parts.path, parts.query, parts.fragment
    rest = url.split("://", 1)[-1].partition("/")[2]

    is_ip = 0
    ip_private = 0
    try:
        ip = ipaddress.ip_address(host)
        is_ip = 1
        ip_private = int(ip.is_private or ip.is_loopback)
    except ValueError:
        if _is_numeric_ip(host):
            is_ip = 1
    sub, dom_label, suffix = ("", host, "") if is_ip else split_host(host)
    domain = f"{dom_label}.{suffix}" if suffix and dom_label else dom_label
    sub_labels = sub.split(".") if sub else []
    host_labels = host.split(".") if host else [""]

    url_len = len(url)
    low = url.lower()
    letter_cnt = sum(c.isalpha() for c in url)
    digit_cnt = sum(c.isdigit() for c in url)
    special_cnt = sum(not c.isalnum() for c in url)
    hd = sum(c.isdigit() for c in host)
    digit_runs = re.findall(r"\d+", url)
    cons_runs = re.findall(r"[^aeiou\W\d_]+", host)
    path_l, query_l, sub_l = path.lower(), query.lower(), sub.lower()
    pq = path_l + "?" + query_l
    path_tokens = [t for t in re.split(r"[/\-_.~=&+]", path + "&" + query) if t]
    qpairs = parse_qsl(query, keep_blank_values=True)

    # ---- brand analysis ----
    brand_in_domain = brand_in_sub = brand_in_path = brand_real = 0
    typo_dist = 4
    homoglyph = 0
    dom_clean = _deleet(dom_label)
    for b in BRAND_NAMES:
        real = any(domain == r or domain.endswith("." + r) for r in BRANDS[b])
        if real:
            brand_real = 1
            continue
        if b in dom_label:
            brand_in_domain = 1
        elif len(b) > 3 and b in dom_clean:
            homoglyph = 1
        if b in sub_l:
            brand_in_sub = 1
        if b in pq:
            brand_in_path = 1
        if dom_label != b and len(b) >= 5 and abs(len(dom_label) - len(b)) <= 2:
            typo_dist = min(typo_dist, _edit_distance(dom_label, b, 3), _edit_distance(dom_clean, b, 3))
    if brand_real:
        typo_dist = 4

    return {
        # ---- original dataset columns ----
        "url_len": url_len,
        "dom_len": len(domain),
        "tld_len": len(suffix),
        "subdom_cnt": len(sub_labels),
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": url.count("="),
        "qm_cnt": url.count("?"),
        "amp_cnt": url.count("&"),
        "dot_cnt": url.count("."),
        "dash_cnt": url.count("-"),
        "under_cnt": url.count("_"),
        "letter_ratio": letter_cnt / url_len if url_len else 0,
        "digit_ratio": digit_cnt / url_len if url_len else 0,
        "spec_ratio": special_cnt / url_len if url_len else 0,
        "is_https": int(low.startswith("https://")),
        "slash_cnt": url.count("/"),
        "entropy": ent(url),
        "path_len": len(path),
        "query_len": len(query),
        # ---- notebook engineered features ----
        "host_len": len(host),
        "host_digits": hd,
        "host_digit_ratio": hd / max(len(host), 1),
        "host_dash": host.count("-"),
        "host_vowel_ratio": sum(c in "aeiou" for c in host) / max(len(host), 1),
        "host_entropy": ent(host),
        "path_entropy": ent(path),
        "query_entropy": ent(query),
        "longest_label": max((len(x) for x in host_labels), default=0),
        "n_labels": len(host_labels),
        "path_depth": path.count("/"),
        "has_www": int(host.startswith("www.")),
        "has_at": int("@" in netloc),
        "double_slash_path": int("//" in rest),
        "pct_enc": url.count("%"),
        "kw_count": sum(k in low for k in KW),
        "kw_in_host": sum(k in host for k in KW),
        "path_dash": path.count("-"),
        "path_under": path.count("_"),
        "path_digits": sum(c.isdigit() for c in path),
        "max_digit_run": max((len(r) for r in digit_runs), default=0),
        "max_cons_run": max((len(r) for r in cons_runs), default=0),
        "file_ext_script": int(bool(re.search(r"\.(php|asp|aspx|jsp|cgi)(\?|$|/)", low))),
        "file_ext_html": int(bool(re.search(r"\.html?(\?|$)", low))),
        "n_params": len(qpairs) if query else 0,
        # ---- NEW: host / domain structure ----
        "is_ip": is_ip,
        "ip_private": ip_private,
        "has_port": int(port is not None),
        "nonstd_port": int(port is not None and port not in (80, 443)),
        "domain_label_len": len(dom_label),
        "domain_label_digits": sum(c.isdigit() for c in dom_label),
        "domain_label_dash": dom_label.count("-"),
        "domain_label_entropy": ent(dom_label),
        "domain_label_vowel_ratio": sum(c in "aeiou" for c in dom_label) / max(len(dom_label), 1),
        "domain_has_digit_letter_mix": int(bool(re.search(r"[a-z]\d|\d[a-z]", dom_label))),
        "sub_len": len(sub),
        "sub_has_dash": int("-" in sub),
        "sub_digits": sum(c.isdigit() for c in sub),
        "sub_has_www_inside": int("www" in sub_labels[1:]) if sub_labels else 0,
        "punycode": int("xn--" in host),
        "non_ascii": int(any(ord(c) > 127 for c in url)),
        "is_free_host": int(domain in FREE_HOSTS or any(host.endswith("." + f) for f in FREE_HOSTS)),
        "is_shortener": int(domain in SHORTENERS),
        "susp_tld": int(suffix in SUSP_TLDS),
        "common_tld": int(suffix in COMMON_TLDS),
        "gov_edu_tld": int(suffix.split(".")[-1] in ("gov", "edu", "mil") or suffix.split(".")[0] in ("gov", "edu", "ac", "nhs")),
        "tld_is_cc": int(len(suffix.split(".")[-1]) == 2),
        "tld_dotted": int("." in suffix),
        # ---- NEW: brand impersonation ----
        "brand_real": brand_real,
        "brand_in_domain": brand_in_domain,
        "brand_in_sub": brand_in_sub,
        "brand_in_path": brand_in_path,
        "brand_homoglyph": homoglyph,
        "typo_dist": typo_dist,
        "brand_any_fake": int(bool(brand_in_domain or brand_in_sub or brand_in_path or homoglyph or typo_dist <= 2)),
        # ---- NEW: path / query ----
        "n_path_tokens": len(path_tokens),
        "avg_token_len": sum(map(len, path_tokens)) / len(path_tokens) if path_tokens else 0,
        "max_token_len": max((len(t) for t in path_tokens), default=0),
        "n_digit_tokens": sum(t.isdigit() for t in path_tokens),
        "n_query_keys": len({k for k, _ in qpairs}),
        "query_val_max_len": max((len(v) for _, v in qpairs), default=0),
        "kw_extra_count": sum(k in low for k in KW_EXTRA),
        "kw_in_sub": sum(k in sub_l for k in KW + KW_EXTRA),
        "kw_in_path": sum(k in path_l for k in KW + KW_EXTRA),
        "kw_in_query": sum(k in query_l for k in KW + KW_EXTRA),
        "upper_ratio_pq": sum(c.isupper() for c in path + query) / max(len(path + query), 1),
        "has_hex_token": int(bool(re.search(r"[0-9a-f]{16,}", low))),
        "long_mixed_token": int(any(len(t) >= 18 and re.search(r"[a-z]", t, re.I) and re.search(r"\d", t) for t in path_tokens)),
        "url_in_path": int(bool(re.search(r"https?(://|%3a%2f%2f)", (path + "?" + query).lower()))),
        "encoded_slash": int("%2f" in low),
        "sensitive_ext": int(bool(SENSITIVE_EXT.search(path_l))),
        "double_ext": int(bool(re.search(r"\.(pdf|doc|docx|jpg|png|txt)\.(php|html?|exe|asp|js)", path_l))),
        "wp_path": int(bool(re.search(r"wp-(content|includes|admin)|/cgi-bin|/~|/\.well-known|/admin", path_l))),
        "ends_slash": int(path.endswith("/")),
        "is_root_only": int(path in ("", "/") and not query and not fragment),
        "fragment_len": len(fragment),
        "path_alpha_ratio": sum(c.isalpha() for c in path) / max(len(path), 1),
        "digits_per_label": hd / max(len(host_labels), 1),
        "special_in_host": sum(not c.isalnum() and c not in ".-" for c in host),
    }


FEATURE_ORDER = list(extract_features_dict("https://www.example.com/a?b=1").keys())


def extract_features(url: str) -> pd.DataFrame:
    return pd.DataFrame([extract_features_dict(url)])
