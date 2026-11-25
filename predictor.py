"""Single place that loads the model and classifies a URL (used by the API and the app)."""
import os
import pickle

from feature_extraction import extract_features, inner_url, is_trusted

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "Final_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(_HERE, "model_scaler.pkl"), "rb") as f:
    _meta = pickle.load(f)

scaler = _meta["scaler"]
feature_order = _meta["feature order"]
threshold = _meta.get("threshold", 0.5)


def phishing_probability(url: str, depth: int = 0) -> float:
    """Probability (0..1) that the link is phishing."""
    inner = inner_url(url)
    if inner and depth < 3:              # redirect link -> judge the real destination
        return phishing_probability(inner, depth + 1)
    if is_trusted(url):                  # big well-known domain (not user-content subdomains)
        return 0.0
    df = extract_features(url)[feature_order]
    return float(model.predict_proba(scaler.transform(df))[0, 1])


def classify(url: str) -> int:
    """1 = phishing, 0 = legitimate."""
    return int(phishing_probability(url) >= threshold)
