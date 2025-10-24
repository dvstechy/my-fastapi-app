from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

model = joblib.load("backend/models/sentiment_model.pkl")
vectorizer = joblib.load("backend/models/tfidf_vectorizer.pkl")

app = FastAPI()

# --- CORS setup ---
origins = [
    "http://localhost:4028",  # React dev server
    "http://127.0.0.1:4028"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackText(BaseModel):
    text: str

@app.post("/predict-sentiment")
def predict_sentiment(feedback: FeedbackText):
    X = vectorizer.transform([feedback.text])
    label = model.predict(X)[0]
    score = max(model.predict_proba(X)[0])
    return {"sentiment_label": label, "sentiment_score": float(score)}
