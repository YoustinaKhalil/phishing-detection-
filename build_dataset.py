"""Build Dataset_expanded.csv

real rows   : the original Dataset.csv (re-featurised with the new extractor)
synthetic   : many extra legit / phishing URL *patterns* so the model sees far more
              than "university homepage vs free-hosting page".

Run:  python build_dataset.py
"""
import random
import string
import sys

import pandas as pd

from feature_extraction import BRANDS, extract_features_dict

SEED = 42
N_SYN_LEGIT = int(sys.argv[1]) if len(sys.argv) > 1 else 90_000
N_SYN_PHISH = int(sys.argv[2]) if len(sys.argv) > 2 else 90_000
rnd = random.Random(SEED)

WORDS = """alpha apex arch atlas aurora beacon bridge bright cedar cloud coast craft crown delta eagle ember
fable falcon forge fusion garden globe grand harbor haven horizon iron island jade journey kite lake lantern
legacy lotus maple meadow metro mint nova oak ocean olive orbit pearl pine pixel planet prime pulse quartz
rapid raven ridge river rocket sage shore silver sky solar spark spring stone summit swift terra thunder tiger
trail urban valley vector velvet vista willow wind zen zephyr data soft tech labs works media group digital
smart global trade travel food health fit home style cars auto bank pay shop store market news daily times
post herald review sport games play music video photo design studio build energy power water green blue red
gold royal first united national city county state school college academy institute clinic dental care
law legal finance capital invest cloud net web app hub link zone world point base core line gate park yard""".split()
SYL = ["ka", "to", "mi", "ra", "ne", "lo", "su", "ve", "da", "pi", "xo", "qu", "zy", "fa", "ge", "hu", "ji", "bo", "wa", "ty"]
SLUG_WORDS = """how to best guide review top new latest update news report story world about contact products
services pricing blog careers team help faq terms privacy policy download install setup tutorial course lesson
article post page category tag archive events gallery video photos recipe travel health sports business tech
science culture opinion politics weather market stock results schedule tickets order cart checkout profile
settings dashboard search compare features overview documentation api reference changelog release notes""".split()
FIRST = "james mary john linda robert patricia michael jennifer william elizabeth david susan ahmed fatima mohamed sara li wei yuki".split()
TOPICS = "python java rust kotlin linux docker kubernetes react vue django flask pandas numpy sql git api cloud security".split()
PAGES_LEGIT = ["", "", "", "about", "contact", "blog", "products", "services", "pricing", "careers", "support/faq", "help",
               "news", "team", "privacy", "terms", "login", "account/settings", "store", "events", "docs", "download"]
SUB_LEGIT = ["www", "www", "www", "blog", "shop", "support", "docs", "app", "api", "mail", "help", "news", "m", "store",
             "portal", "learn", "static", "cdn", "dev", "status", "careers", "investors", "en", "us", "eu"]
GOOD_TLDS = ["com"] * 40 + ["org"] * 10 + ["net"] * 6 + ["edu"] * 3 + ["gov"] * 2 + ["io"] * 4 + ["co"] * 2 + ["info"] * 1 + \
            ["co.uk", "de", "fr", "it", "es", "nl", "ca", "com.au", "co.jp", "jp", "ru", "pl", "se", "no", "ch", "be", "at", "dk",
             "fi", "pt", "cz", "br", "com.br", "in", "co.in", "cn", "com.cn", "kr", "tw", "com.mx", "co.za", "eg", "com.eg", "sa",
             "ae", "tr", "com.tr", "id", "co.id", "vn", "th", "my", "sg", "nz", "co.nz", "ie", "gr", "hu", "ro", "ua", "tv", "me",
             "ai", "app", "dev", "tech", "online", "store", "shop", "site", "xyz", "club"]
GOOD_TLDS_W = GOOD_TLDS
BAD_TLDS = ["xyz", "top", "icu", "click", "buzz", "cfd", "sbs", "live", "cc", "tk", "ml", "ga", "cf", "gq", "work", "support",
            "rest", "monster", "lol", "cyou", "bond", "quest", "vip", "shop", "site", "online", "store", "link", "win", "bid",
            "loan", "date", "pw", "ws", "su", "info", "biz", "fun", "life", "world", "today", "space", "website", "com", "net",
            "org", "com", "com", "co", "me", "us", "in", "ru", "de", "pl", "br"]
FREE = ["pages.dev", "weebly.com", "github.io", "web.app", "firebaseapp.com", "wixsite.com", "blogspot.com", "glitch.me",
        "netlify.app", "000webhostapp.com", "duckdns.org", "workers.dev", "vercel.app", "herokuapp.com", "r2.dev",
        "webflow.io", "godaddysites.com", "myshopify.com", "square.site", "carrd.co", "onrender.com", "notion.site",
        "repl.co", "azurewebsites.net", "appspot.com", "ddns.net", "hopto.org", "trycloudflare.com", "surge.sh"]
SHORT = ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "qrco.de", "q-r.to", "shorturl.at", "tiny.cc", "s.id", "t.ly", "ow.ly"]
PHISH_KW = ["login", "signin", "secure", "account", "update", "verify", "confirm", "webscr", "support", "wallet", "password",
            "billing", "invoice", "payment", "suspended", "unlock", "recover", "restore", "claim", "reward", "security",
            "alert", "validate", "customer", "helpdesk", "limited", "official", "auth", "kyc", "refund"]
PHISH_PATH = ["login.php", "signin.html", "verify/account", "secure/update", "webscr", "index.php", "auth/login", "account/confirm",
              "wp-content/uploads/login.php", "wp-includes/js/signin.html", "cgi-bin/login.pl", "~user/secure/index.html",
              "admin/js/verify.php", "update/billing", "client/authentication", "myaccount/signin", "online/banking/login",
              "app/recover-password", "support/ticket", "claim/reward", "office/auth", "docs/shared/view", "invoice/pay"]
BRAND_LIST = sorted(BRANDS)
BRAND_DOMAINS = {b: sorted(d)[0] for b, d in BRANDS.items()}

BIG_LEGIT = [
    ("wikipedia.org", "en.wikipedia.org/wiki/{Title}"), ("github.com", "github.com/{user}/{repo}/blob/main/{file}"),
    ("github.com", "github.com/{user}/{repo}/issues/{n}"), ("stackoverflow.com", "stackoverflow.com/questions/{n}/{slug}"),
    ("amazon.com", "www.amazon.com/{Slug}/dp/{asin}"), ("amazon.com", "www.amazon.com/dp/{asin}?ref=nav_{w}&th=1"),
    ("youtube.com", "www.youtube.com/watch?v={vid}"), ("youtube.com", "www.youtube.com/watch?v={vid}&list={lst}&index={d}"),
    ("reddit.com", "www.reddit.com/r/{w}/comments/{id6}/{slug}/"), ("medium.com", "medium.com/@{user}/{slug}-{hex12}"),
    ("linkedin.com", "www.linkedin.com/in/{user}-{hex8}"), ("x.com", "x.com/{user}/status/{n19}"),
    ("twitter.com", "twitter.com/{user}/status/{n19}?s=20"), ("facebook.com", "www.facebook.com/{user}/posts/{n15}"),
    ("instagram.com", "www.instagram.com/p/{id11}/"), ("google.com", "www.google.com/search?q={q}&hl=en&source=hp&ei={id20}"),
    ("google.com", "maps.google.com/maps?q={q}&ll={lat},{lon}&z=12"), ("google.com", "accounts.google.com/signin/v2/identifier?service=mail&continue=https%3A%2F%2Fmail.google.com"),
    ("microsoft.com", "support.microsoft.com/en-us/office/{slug}-{guid}"), ("microsoft.com", "learn.microsoft.com/en-us/{w}/{slug}"),
    ("live.com", "login.live.com/login.srf?wa=wsignin1.0&rpsnv=13&ct={n10}"), ("microsoftonline.com", "login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id={guid}&response_type=code&scope=openid%20profile"),
    ("apple.com", "support.apple.com/en-us/{id8}"), ("apple.com", "www.apple.com/iphone-{d}/specs/"), ("paypal.com", "www.paypal.com/signin"),
    ("paypal.com", "www.paypal.com/myaccount/summary"), ("ebay.com", "www.ebay.com/itm/{n12}?hash=item{hex10}"), ("ebay.com", "signin.ebay.com/ws/eBayISAPI.dll?SignIn&ru=https%3A%2F%2Fwww.ebay.com"),
    ("netflix.com", "www.netflix.com/title/{n8}"), ("netflix.com", "www.netflix.com/login"), ("spotify.com", "open.spotify.com/track/{id22}?si={hex16}"),
    ("dropbox.com", "www.dropbox.com/s/{id15}/{file}?dl=0"), ("zoom.us", "us02web.zoom.us/j/{n11}?pwd={id32}"), ("bbc.com", "www.bbc.com/news/{slug}-{n8}"),
    ("bbc.co.uk", "www.bbc.co.uk/sport/football/{n8}"), ("nytimes.com", "www.nytimes.com/{yr}/{mo}/{dy}/{w}/{slug}.html"),
    ("theguardian.com", "www.theguardian.com/{w}/{yr}/{mon}/{dy}/{slug}"), ("cnn.com", "www.cnn.com/{yr}/{mo}/{dy}/{w}/{slug}/index.html"),
    ("bankofamerica.com", "secure.bankofamerica.com/login/sign-in/signOnV2Screen.go"), ("chase.com", "secure.chase.com/web/auth/dashboard"),
    ("wellsfargo.com", "connect.secure.wellsfargo.com/auth/login/present?origin=cob"), ("hsbc.co.uk", "www.hsbc.co.uk/accounts/products/{slug}/"),
    ("adobe.com", "helpx.adobe.com/{w}/{slug}.html"), ("mozilla.org", "developer.mozilla.org/en-US/docs/Web/{w}/{Slug}"),
    ("python.org", "docs.python.org/3/library/{w}.html#{slug}"), ("cloudflare.com", "www.cloudflare.com/learning/{w}/{slug}/"),
    ("aliexpress.com", "www.aliexpress.com/item/{n16}.html?spm=a2g0o.productlist.main.{d}.{hex8}"), ("walmart.com", "www.walmart.com/ip/{Slug}/{n9}"),
    ("booking.com", "www.booking.com/hotel/{w}/{slug}.en-gb.html?aid={n6}&label=gen173nr"), ("imdb.com", "www.imdb.com/title/tt{n7}/?ref_=nv_sr_srsg_{d}"),
    ("w3schools.com", "www.w3schools.com/{w}/{w2}_intro.asp"), ("jsdelivr.net", "cdn.jsdelivr.net/npm/{w}@{d}.{d}.{d}/dist/{w}.min.js"),
    ("cloudflare.com", "cdnjs.cloudflare.com/ajax/libs/{w}/{d}.{d}.{d}/{w}.min.js"), ("sciencedirect.com", "www.sciencedirect.com/science/article/pii/S{n16}"),
    ("nih.gov", "pubmed.ncbi.nlm.nih.gov/{n8}/"), ("arxiv.org", "arxiv.org/abs/{yr2}{mo}.{n5}"), ("who.int", "www.who.int/news-room/fact-sheets/detail/{slug}"),
    ("un.org", "www.un.org/en/{w}/{slug}"), ("europa.eu", "ec.europa.eu/info/{w}/{slug}_en"), ("gov.uk", "www.gov.uk/{slug}/{w}"),
    ("harvard.edu", "www.harvard.edu/{w}/{slug}/"), ("mit.edu", "news.mit.edu/{yr}/{slug}-{mo}{dy}"), ("coursera.org", "www.coursera.org/learn/{slug}?specialization={slug}"),
    ("slack.com", "app.slack.com/client/T{id8}/C{id8}"), ("notion.so", "www.notion.so/{slug}-{hex32}"), ("trello.com", "trello.com/c/{id8}/{d}-{slug}"),
    ("shopify.com", "www.shopify.com/blog/{slug}"), ("etsy.com", "www.etsy.com/listing/{n9}/{slug}?ref=sr_gallery-{d}-{d}"),
    ("tripadvisor.com", "www.tripadvisor.com/Hotel_Review-g{n6}-d{n6}-Reviews-{Slug}.html"), ("quora.com", "www.quora.com/{Slug}"),
    ("pinterest.com", "www.pinterest.com/pin/{n18}/"), ("twitch.tv", "www.twitch.tv/{user}"), ("tiktok.com", "www.tiktok.com/@{user}/video/{n19}"),
    ("whatsapp.com", "faq.whatsapp.com/{n15}/?lang=en"), ("telegram.org", "telegram.org/faq#{w}"), ("outlook.com", "outlook.live.com/mail/0/inbox/id/{id32}"),
    ("office.com", "www.office.com/launch/word?auth=2&from=Marketing"), ("steampowered.com", "store.steampowered.com/app/{n6}/{Slug}/"),
    ("binance.com", "www.binance.com/en/trade/BTC_USDT?theme=dark&type=spot"), ("coinbase.com", "www.coinbase.com/price/{w}"),
]
PROTO_FIX = {"http": 6, "https": 94}


def r_hex(n): return "".join(rnd.choice("0123456789abcdef") for _ in range(n))
def r_alnum(n, up=True):
    pool = string.ascii_lowercase + string.digits + (string.ascii_uppercase if up else "")
    return "".join(rnd.choice(pool) for _ in range(n))
def r_num(n): return str(rnd.randint(10 ** (n - 1), 10 ** n - 1))
def word(): return rnd.choice(WORDS)
def slug(k=None): return "-".join(rnd.choice(SLUG_WORDS + WORDS) for _ in range(k or rnd.randint(2, 7)))
def name_label():
    r = rnd.random()
    if r < 0.45: return word() + word()
    if r < 0.60: return word() + "-" + word()
    if r < 0.80: return word()
    if r < 0.90: return "".join(rnd.choice(SYL) for _ in range(rnd.randint(2, 4)))
    return word() + rnd.choice(["24", "365", "360", "2go", "1", "123", "x", "hq"])
def guid(): return f"{r_hex(8)}-{r_hex(4)}-{r_hex(4)}-{r_hex(4)}-{r_hex(12)}"


def fill(tpl: str) -> str:
    rep = {
        "Title": lambda: "_".join(w.capitalize() for w in slug(rnd.randint(1, 4)).split("-")), "user": lambda: rnd.choice(FIRST) + rnd.choice(["", str(rnd.randint(1, 99)), "_" + word()]),
        "repo": lambda: slug(rnd.randint(1, 3)), "file": lambda: rnd.choice(["README.md", "src/main.py", "index.js", "setup.py", "docs/guide.pdf", "report.pdf"]),
        "n": lambda: r_num(rnd.randint(4, 8)), "slug": slug, "Slug": lambda: "-".join(w.capitalize() for w in slug().split("-")),
        "asin": lambda: "B0" + r_alnum(8).upper(), "w2": lambda: rnd.choice(WORDS), "w": lambda: rnd.choice(TOPICS + WORDS), "vid": lambda: r_alnum(11),
        "lst": lambda: "PL" + r_alnum(30), "d": lambda: str(rnd.randint(1, 20)), "id6": lambda: r_alnum(6, False), "id8": lambda: r_alnum(8), "id11": lambda: r_alnum(11),
        "id15": lambda: r_alnum(15, False), "id20": lambda: r_alnum(20), "id22": lambda: r_alnum(22), "id32": lambda: r_alnum(32), "hex8": lambda: r_hex(8),
        "hex10": lambda: r_hex(10), "hex12": lambda: r_hex(12), "hex16": lambda: r_hex(16), "hex32": lambda: r_hex(32), "guid": guid,
        "n5": lambda: r_num(5), "n6": lambda: r_num(6), "n7": lambda: r_num(7), "n8": lambda: r_num(8), "n9": lambda: r_num(9), "n10": lambda: r_num(10),
        "n11": lambda: r_num(11), "n12": lambda: r_num(12), "n15": lambda: r_num(15), "n16": lambda: r_num(16), "n18": lambda: r_num(18), "n19": lambda: r_num(19),
        "yr": lambda: str(rnd.randint(2015, 2026)), "yr2": lambda: str(rnd.randint(15, 26)), "mo": lambda: f"{rnd.randint(1, 12):02d}", "dy": lambda: f"{rnd.randint(1, 28):02d}",
        "mon": lambda: rnd.choice(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
        "q": lambda: "+".join(rnd.choice(SLUG_WORDS + WORDS) for _ in range(rnd.randint(1, 5))), "lat": lambda: f"{rnd.uniform(-60, 60):.4f}", "lon": lambda: f"{rnd.uniform(-170, 170):.4f}",
    }
    out = tpl
    for k, fn in rep.items():
        while "{" + k + "}" in out:
            out = out.replace("{" + k + "}", fn(), 1)
    return out


def proto(p_https=0.93): return "https://" if rnd.random() < p_https else "http://"


# --------------------------------------------------------------------------- legit
def legit_company():
    d = name_label() + "." + rnd.choice(GOOD_TLDS_W)
    sub = rnd.choice(SUB_LEGIT) if rnd.random() < 0.8 else ""
    host = (sub + "." if sub else "") + d
    if rnd.random() < 0.03: host = f"{rnd.choice(SUB_LEGIT)}.{rnd.choice(SUB_LEGIT)}.{d}"
    r = rnd.random()
    if r < 0.28: path = ""
    elif r < 0.55: path = "/" + rnd.choice(PAGES_LEGIT)
    elif r < 0.80: path = "/" + "/".join([rnd.choice(["products", "blog", "news", "category", "shop", "articles", "support", "docs", "en", "services", "resources"]), slug()])
    elif r < 0.92: path = f"/{rnd.choice(['blog', 'news', 'article'])}/{rnd.randint(2015, 2026)}/{rnd.randint(1, 12):02d}/{slug()}"
    else: path = "/" + "/".join(slug(rnd.randint(1, 3)) for _ in range(rnd.randint(3, 7)))
    if rnd.random() < 0.12 and path: path += rnd.choice([".html", ".htm", ".php", ".aspx", ".jsp", "/index.html", "/"])
    q = ""
    if rnd.random() < 0.22:
        keys = rnd.sample(["id", "page", "q", "sort", "ref", "utm_source", "utm_medium", "utm_campaign", "lang", "category", "tab", "v", "order", "filter", "p"], rnd.randint(1, 5))
        q = "?" + "&".join(f"{k}={rnd.choice([r_num(rnd.randint(1, 6)), word(), slug(2), r_alnum(rnd.randint(4, 12), False)])}" for k in keys)
    port = f":{rnd.choice([8080, 8443, 3000, 8000])}" if rnd.random() < 0.01 else ""
    return proto() + host + port + path + q


def legit_big():
    dom, tpl = rnd.choice(BIG_LEGIT)
    u = proto(0.97) + fill(tpl)
    if rnd.random() < 0.08: u += rnd.choice(["&utm_source=newsletter&utm_medium=email", "?utm_campaign=" + slug(2), "#" + word()])
    return u


def legit_gov_edu():
    k = rnd.random()
    n = name_label()
    if k < 0.35: host = f"www.{n}.edu"
    elif k < 0.55: host = f"{rnd.choice(['www', 'admissions', 'library', 'research', 'www2'])}.{n}.ac.{rnd.choice(['uk', 'jp', 'in', 'id', 'za', 'nz'])}"
    elif k < 0.75: host = f"www.{n}.gov{rnd.choice(['', '.uk', '.au', '.cn', '.br', '.eg', '.in'])}"
    elif k < 0.9: host = f"www.{n}.{rnd.choice(['gouv.fr', 'gob.mx', 'gov.it', 'go.jp', 'admin.ch', 'bund.de'])}"
    else: host = f"{n}.org"
    path = "" if rnd.random() < .3 else "/" + "/".join(slug(rnd.randint(1, 3)) for _ in range(rnd.randint(1, 5))) + rnd.choice(["", ".html", ".pdf", "/", ".aspx"])
    return proto(0.85) + host + path


def legit_local():
    host = rnd.choice(["localhost", "127.0.0.1", f"192.168.{rnd.randint(0, 5)}.{rnd.randint(1, 254)}", f"10.0.{rnd.randint(0, 5)}.{rnd.randint(1, 254)}", f"172.16.{rnd.randint(0, 5)}.{rnd.randint(1, 254)}"])
    return rnd.choice(["http://", "https://"]) + host + rnd.choice(["", ":3000", ":8080", ":5000", ":8000"]) + rnd.choice(["", "/", "/admin", "/login", "/dashboard", "/api/v1/users", "/index.html"])


def legit_ugc():
    """Legit pages hosted on free platforms / user-content sites (hard negatives)."""
    host = rnd.choice(["github.io", "wordpress.com", "blogspot.com", "medium.com", "netlify.app", "vercel.app", "herokuapp.com", "readthedocs.io", "substack.com", "gitlab.io", "web.app", "pages.dev", "weebly.com", "wixsite.com", "myshopify.com"])
    sub = name_label()
    path = "" if rnd.random() < 0.4 else "/" + "/".join(slug(rnd.randint(1, 3)) for _ in range(rnd.randint(1, 3))) + rnd.choice(["", "/", ".html"])
    return "https://" + sub + "." + host + path


def legit_brand_keywords():
    """Legit sites containing 'scary' words (login, secure, bank...) but on their own domain."""
    n = name_label()
    tld = rnd.choice(["com", "com", "org", "net", "co.uk", "de", "bank", "com.au"])
    host = rnd.choice(["secure", "login", "accounts", "my", "online", "portal", "support", "billing", "pay", "auth", "id"]) + "." + n + "." + tld
    path = "/" + rnd.choice(["login", "signin", "account/login", "user/password-reset", "verify-email", "support/ticket/new", "billing/invoices", "customer/update-profile", "auth/callback"]) + rnd.choice(["", "", "?next=/dashboard", "?lang=en", "?redirect=%2Fhome"])
    return "https://" + host + path


def legit_shortener():
    return "https://" + rnd.choice(SHORT[:6]) + "/" + r_alnum(rnd.randint(5, 8))  # shorteners alone are NOT enough to flag; some legit


# --------------------------------------------------------------------------- phishing
def brand_domain_fake(b):
    r = rnd.random()
    kw = rnd.choice(PHISH_KW)
    if r < 0.18:
        n = b + "-" + kw
    elif r < 0.33:
        n = kw + "-" + b
    elif r < 0.45:
        n = b + kw
    elif r < 0.62:  # typo / homoglyph
        i = rnd.randrange(len(b))
        mode = rnd.random()
        if mode < 0.3: n = b[:i] + b[i + 1:]
        elif mode < 0.5: n = b[:i] + b[i] * 2 + b[i + 1:]
        elif mode < 0.75: n = b.translate(str.maketrans({"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}))
        elif mode < 0.9: n = b.replace("m", "rn", 1) if "m" in b else b + "s"
        else: n = b[:i] + rnd.choice(string.ascii_lowercase) + b[i + 1:]
        if n == b: n = b + "-" + kw
    elif r < 0.75:
        n = b + "-" + kw + "-" + rnd.choice(["center", "team", "help", "service", "care", "online", str(rnd.randint(1, 999))])
    elif r < 0.88:
        n = b + str(rnd.randint(1, 9999))
    else:
        n = kw + b + "-" + name_label()
    return n


def phish_brand():
    b = rnd.choice(BRAND_LIST)
    r = rnd.random()
    kw = rnd.choice(PHISH_KW)
    path = "/" + rnd.choice(PHISH_PATH) if rnd.random() < 0.75 else ""
    if rnd.random() < 0.35 and path: path += "?" + rnd.choice(["id", "session", "token", "user", "ref"]) + "=" + r_alnum(rnd.randint(8, 40), False)
    if r < 0.40:   # brand-ish domain
        host = brand_domain_fake(b) + "." + rnd.choice(BAD_TLDS)
        if rnd.random() < 0.3: host = rnd.choice(["www", "secure", "login", "my", "account"]) + "." + host
    elif r < 0.65:  # brand as subdomain of unrelated domain  paypal.com.evil.xyz
        evil = name_label() + "." + rnd.choice(BAD_TLDS)
        host = rnd.choice([f"{b}.com.{evil}", f"{b}.{kw}.{evil}", f"www.{b}.com.{kw}.{evil}", f"{kw}-{b}.{evil}", f"{b}-{kw}.{evil}", f"{kw}.{b}.{evil}"])
    elif r < 0.80:  # brand in path of unrelated (compromised) site
        host = rnd.choice(["www.", "", "shop."]) + name_label() + "." + rnd.choice(GOOD_TLDS_W[:60] + BAD_TLDS[:15])
        path = "/" + rnd.choice(["wp-content", "wp-includes", "images", "includes", "admin", "cgi-bin", "~" + word()]) + "/" + b + "/" + rnd.choice(PHISH_PATH)
    elif r < 0.92:  # free hosting with brand
        host = rnd.choice([b + "-" + kw, kw + "-" + b, b + "-" + name_label(), b + kw + str(rnd.randint(1, 99))]) + "." + rnd.choice(FREE)
    else:           # @ trick
        host = f"{b}.com@{name_label()}.{rnd.choice(BAD_TLDS)}"
    return rnd.choice(["https://", "https://", "https://", "http://"]) + host + path


def phish_ip():
    k = rnd.random()
    if k < 0.75: ip = ".".join(str(rnd.randint(1, 254)) for _ in range(4))
    elif k < 0.9: ip = str(rnd.randint(1_000_000_000, 4_000_000_000))
    else: ip = "0x" + r_hex(8)
    port = rnd.choice(["", "", ":8080", ":2083", ":8443", ":81", ":8888"])
    path = "/" + rnd.choice(["~" + word() + "/", "", "wp-admin/", "login/", "auth/", "secure/"]) + rnd.choice(["login.php", "index.html", "verify.php", "update/", "bank/", "id=" + r_num(6)])
    return rnd.choice(["http://", "http://", "http://", "https://"]) + ip + port + path


def phish_free_host():
    n = rnd.choice([name_label() + "-" + rnd.choice(PHISH_KW), rnd.choice(PHISH_KW) + "-" + name_label(), r_alnum(rnd.randint(6, 14), False), name_label() + r_num(rnd.randint(2, 5)), "-".join(word() for _ in range(rnd.randint(2, 4)))])
    host = n + "." + rnd.choice(FREE)
    path = rnd.choice(["", "/", "/" + rnd.choice(PHISH_PATH), "/" + slug(2), "/" + r_alnum(rnd.randint(6, 12), False), "/" + rnd.choice(["index.html", "home", "view", "form", "login"])])
    return "https://" + host + path


def phish_shortener():
    return "https://" + rnd.choice(SHORT) + "/" + r_alnum(rnd.randint(5, 8))


def phish_dga():
    n = "".join(rnd.choice(string.ascii_lowercase + string.digits) for _ in range(rnd.randint(6, 20)))
    if rnd.random() < 0.5: n = "".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(8, 20)))
    host = n + "." + rnd.choice(BAD_TLDS[:30])
    if rnd.random() < 0.3: host = r_alnum(rnd.randint(3, 8), False) + "." + host
    path = rnd.choice(["", "/", "/" + r_alnum(rnd.randint(6, 14), False) + "/" + r_alnum(rnd.randint(4, 8), False), "/" + r_alnum(8, False) + ".php?id=" + r_hex(rnd.randint(12, 40)), "/" + r_hex(16)])
    return rnd.choice(["https://", "http://"]) + host + path


def phish_compromised():
    """Normal-looking domains with webshell/kit style paths and NO brand words."""
    host = rnd.choice(["www.", "", "blog."]) + name_label() + "." + rnd.choice(GOOD_TLDS_W[:70] + BAD_TLDS[:10])
    path = "/" + rnd.choice(["wp-content/uploads/" + str(rnd.randint(2019, 2025)) + "/" + r_num(2), "wp-includes/" + word(), "images/" + word(), "assets/" + r_alnum(6, False), "cgi-bin", "components/" + word(), "tmp", "sites/default/files", "~" + word(), "cache/" + r_hex(8)])
    path += "/" + rnd.choice(["index.php", "login.php", "verify.html", "doc.html", "view.php?id=" + r_hex(rnd.randint(10, 32)), "secure/", "mail/", "form.php", "o365/", "adobe/pdf/", "dropbox/", "file." + rnd.choice(["pdf", "doc"]) + ".php", r_alnum(rnd.randint(6, 14), False) + ".html"])
    if rnd.random() < 0.25: path += "?" + rnd.choice(["email", "user", "id", "session", "key"]) + "=" + rnd.choice([r_alnum(rnd.randint(8, 30), False), r_num(rnd.randint(5, 12)), name_label() + "%40" + name_label() + ".com", "dXNlcg" + r_alnum(10, False) + "=="])
    return rnd.choice(["https://", "http://", "http://"]) + host + path


def phish_redirect():
    target = "https://" + name_label() + "." + rnd.choice(BAD_TLDS) + "/" + rnd.choice(PHISH_PATH)
    host = rnd.choice(["www.", ""]) + name_label() + "." + rnd.choice(GOOD_TLDS_W[:40])
    enc = rnd.choice([target, target.replace(":", "%3A").replace("/", "%2F")])
    return "https://" + host + "/" + rnd.choice(["redirect", "out", "r", "click", "go", "link", "url"]) + "?" + rnd.choice(["url", "u", "redirect", "to", "target", "next", "goto"]) + "=" + enc


def phish_longtail():
    host_core = rnd.choice(["secure-" + name_label(), name_label() + "-secure", "my-" + name_label(), name_label() + "-" + rnd.choice(PHISH_KW) + "-" + r_num(rnd.randint(2, 4)), name_label() + "." + rnd.choice(PHISH_KW)])
    host = host_core + "." + rnd.choice(BAD_TLDS)
    path = "/" + "/".join(rnd.choice(PHISH_KW + [r_alnum(rnd.randint(5, 15), False)]) for _ in range(rnd.randint(1, 5))) + rnd.choice(["", ".php", ".html", "/"])
    if rnd.random() < 0.4: path += "?" + "&".join(f"{rnd.choice(['a', 'sid', 'token', 'cmd', 'dispatch', 'session', 'id'])}={r_alnum(rnd.randint(10, 60), False)}" for _ in range(rnd.randint(1, 3)))
    return rnd.choice(["https://", "https://", "http://"]) + host + path


def phish_punycode_crypto():
    if rnd.random() < 0.4:
        host = "xn--" + r_alnum(rnd.randint(6, 12), False) + "-" + rnd.choice(["9ua", "bya", "cua", "k2a", "r1a"]) + "." + rnd.choice(["com", "net", "xyz", "org", "top"])
    else:
        w = rnd.choice(["airdrop", "claim", "free-bitcoin", "nft-mint", "wallet-connect", "giveaway", "reward", "binance-claim", "eth-bonus", "usdt-rewards", "walletvalidate", "dapp-sync"])
        host = w + "-" + name_label() + "." + rnd.choice(BAD_TLDS[:25])
    return "https://" + host + rnd.choice(["", "/", "/connect", "/claim?ref=" + r_alnum(8, False), "/wallet/validate", "/mint"])


LEGIT_GENS = [(legit_company, 33), (legit_big, 32), (legit_gov_edu, 10), (legit_local, 2), (legit_ugc, 10), (legit_brand_keywords, 10), (legit_shortener, 3)]
PHISH_GENS = [(phish_brand, 28), (phish_ip, 7), (phish_free_host, 16), (phish_shortener, 4), (phish_dga, 11), (phish_compromised, 15), (phish_redirect, 5), (phish_longtail, 9), (phish_punycode_crypto, 5)]


def gen(n, gens, label):
    fns, w = zip(*gens)
    seen, out = set(), []
    while len(out) < n:
        u = rnd.choices(fns, w)[0]()
        if u in seen:
            continue
        seen.add(u)
        out.append((u, label, "synthetic"))
    return out


def main():
    real = pd.read_csv("Dataset.csv").dropna(subset=["tld"])
    real = real[["url", "label"]].drop_duplicates()
    conflict = real.groupby("url")["label"].nunique()
    real = real[~real.url.isin(conflict[conflict > 1].index)].drop_duplicates("url")
    real = real[real.url.str.len() < 2000].assign(source="real")
    print("real rows:", len(real), real.label.value_counts().to_dict())

    syn = gen(N_SYN_LEGIT, LEGIT_GENS, 0) + gen(N_SYN_PHISH, PHISH_GENS, 1)
    syn = pd.DataFrame(syn, columns=["url", "label", "source"])
    syn = syn[~syn.url.isin(set(real.url))]
    print("synthetic rows:", len(syn), syn.label.value_counts().to_dict())

    all_df = pd.concat([real, syn], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    feats = pd.DataFrame([extract_features_dict(u) for u in all_df.url])
    out = pd.concat([all_df, feats], axis=1)
    out.to_csv("Dataset_expanded.csv", index=False)
    print("saved Dataset_expanded.csv", out.shape)


if __name__ == "__main__":
    main()
