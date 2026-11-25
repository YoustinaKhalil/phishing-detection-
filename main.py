from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_extraction import INVALID_URL_MESSAGE, is_valid_url
from predictor import classify, phishing_probability

app = FastAPI(title="Phishing Website Detection API")


class InputData(BaseModel):
    url: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    url = data.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty")
    if not is_valid_url(url):
        raise HTTPException(status_code=422, detail=INVALID_URL_MESSAGE)
    try:
        prob = phishing_probability(url)
        prediction = classify(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {
        "url": url,
        "result": "Phishing Website" if prediction == 1 else "Legitimate Website",
        "phishing_probability": round(prob, 4),
    }
