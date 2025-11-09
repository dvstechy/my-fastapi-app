from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, os

app = FastAPI()

# --- CORS setup ---
origins = [
    "https://heritage-bites.vercel.app",
    "http://127.0.0.1:4028",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define request model ---
class FeedbackText(BaseModel):
    text: str
    user_type: str  # "User" or "Farmer"

# --- Helper: load model + vectorizer based on user_type ---
def load_role_model(user_type: str):
    user_type = user_type.lower()
    if user_type == "user":
        model_path = "backend/models/sentiment_model_user.pkl"
        vectorizer_path = "backend/models/tfidf_vectorizer_user.pkl"
    elif user_type == "farmer":
        model_path = "backend/models/sentiment_model_farmer.pkl"
        vectorizer_path = "backend/models/tfidf_vectorizer_farmer.pkl"
    else:
        raise ValueError("Invalid user_type. Must be 'User' or 'Farmer'.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# --- Prediction Endpoint ---
@app.post("/predict-sentiment")
def predict_sentiment(feedback: FeedbackText):
    try:
        model, vectorizer = load_role_model(feedback.user_type)
    except Exception as e:
        return {"error": str(e)}

    X = vectorizer.transform([feedback.text])
    probs = model.predict_proba(X)[0]
    label = model.predict(X)[0]

    # Find correct score
    label_index = list(model.classes_).index(label)
    score = float(probs[label_index])

    # Neutral buffer
    if 0.45 < score < 0.55:
        label = "neutral"

    return {
        "sentiment_label": label,
        "sentiment_score": round(score, 4),
        "model_used": feedback.user_type.capitalize()
    }
