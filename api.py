from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# --- Load model & vectorizer ---
model = joblib.load("backend/models/sentiment_model.pkl")
vectorizer = joblib.load("backend/models/tfidf_vectorizer.pkl")

app = FastAPI()

# --- CORS setup ---
origins = [
    "https://heritage-bites.vercel.app",  # React dev server
    "http://127.0.0.1:4028"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackText(BaseModel):
    text: str

@app.post("/predict-sentiment")
def predict_sentiment(feedback: FeedbackText):
    X = vectorizer.transform([feedback.text])
    probs = model.predict_proba(X)[0]
    label = model.predict(X)[0]

    # Find the correct score for the predicted label
    label_index = list(model.classes_).index(label)
    score = float(probs[label_index])

    # Optional: Apply neutral buffer for uncertain predictions
    if 0.45 < score < 0.55:
        label = "neutral"

    return {
        "sentiment_label": label,
        "sentiment_score": round(score, 4)
    }
