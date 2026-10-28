import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account", "bank", "confirm",
                    "signin", "password", "paypal", "free", "bonus", "ebay", "webscr"]

FEATURE_NAMES = [
    "url_length", "num_dots", "num_hyphens", "num_digits", "has_https",
    "num_at", "num_slashes", "num_question", "num_equals", "num_underscore",
    "has_ip", "domain_length", "num_subdomains", "path_length",
    "suspicious_words", "digit_ratio", "special_ratio",
]


def extract_url_features(url):
    """Convert one URL into a list of numeric features (same order as FEATURE_NAMES)."""
    url = str(url).strip()
    uses_https = 1 if url.lower().startswith("https://") else 0
    # The training URLs are stored without "http://" / "https://", so the scheme is removed
    # before measuring the URL; whether HTTPS was used is kept as a separate feature.
    url = re.sub(r"^[a-zA-Z]+://", "", url)
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        domain = parsed.netloc.lower()
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
    except ValueError:                                  # malformed URL (e.g. broken brackets)
        rest = url.split("://", 1)[-1]
        domain, _, path = rest.partition("/")
        domain = domain.lower()
    length = max(len(url), 1)

    features = []
    features.append(len(url))                                    # length of URL
    features.append(url.count("."))                              # number of dots
    features.append(url.count("-"))                              # number of hyphens
    features.append(sum(c.isdigit() for c in url))               # number of digits
    features.append(uses_https)                                  # uses https or not
    features.append(url.count("@"))                              # @ symbol
    features.append(url.count("/"))                              # slashes
    features.append(url.count("?"))                              # query marks
    features.append(url.count("="))                              # parameters
    features.append(url.count("_"))                              # underscores
    features.append(1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", domain) else 0)  # IP address as host
    features.append(len(domain))                                 # domain length
    features.append(max(domain.count(".") - 1, 0))               # number of sub-domains
    features.append(len(path))                                   # path + query length
    features.append(sum(w in url.lower() for w in SUSPICIOUS_WORDS))  # suspicious words
    features.append(sum(c.isdigit() for c in url) / length)      # digit ratio
    features.append(sum(not c.isalnum() for c in url) / length)  # special-character ratio

    return features
