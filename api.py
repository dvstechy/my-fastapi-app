from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, re, numpy as np

app = FastAPI()

# --- CORS setup ---
origins = [
    "https://heritage-bites.vercel.app",
    "http://localhost:4028",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request model with optional fields ---
class FeedbackText(BaseModel):
    user_type: str  # "User" or "Farmer"
    text: str = ""  # fallback single text
    e_market_review: str = ""
    recipe_review: str = ""
    chatbot_review: str = ""
    contribution_review: str = ""
    overall_review: str = ""
    e_market_rating: int = None
    recipe_rating: int = None
    chatbot_rating: int = None
    contribution_rating: int = None
    overall_rating: int = None

# --- Load model + vectorizer ---
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

# --- Clean text function ---
def clean_text(text: str):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# --- Prediction endpoint ---
@app.post("/predict-sentiment")
def predict_sentiment(feedback: FeedbackText):
    try:
        model, vectorizer = load_role_model(feedback.user_type)
    except Exception as e:
        return {"error": str(e)}

    # --- Combine reviews safely ---
    combined_text = " ".join([
        feedback.text,
        feedback.e_market_review,
        feedback.recipe_review,
        feedback.chatbot_review,
        feedback.contribution_review,
        feedback.overall_review
    ])
    cleaned_text = clean_text(combined_text)
    X = vectorizer.transform([cleaned_text])

    # --- Predict ---
    probs = model.predict_proba(X)[0]
    label = model.predict(X)[0]

    label_index = list(model.classes_).index(label)
    text_score = float(probs[label_index])
  # text-based probability

    # --- Use ratings if provided to adjust label ---
    ratings = [
        feedback.e_market_rating,
        feedback.recipe_rating,
        feedback.chatbot_rating,
        feedback.contribution_rating,
        feedback.overall_rating
    ]
    ratings = [r for r in ratings if r is not None]

    # --- Use ratings to influence the score (without overwriting model prediction) ---
    if ratings:
        avg_rating = np.mean(ratings)
    # Convert average rating (1-5) to 0-1 scale
        rating_score = (avg_rating - 1) / 4  # 1 → 0, 5 → 1
    # Combine with text-based score (weighted average)
        combined_score = 0.7 * score + 0.3 * rating_score
    else:
        final_score = text_score

# Determine final label based on combined_score thresholds
    if final_score < 0.45:
        label = "negative"
    elif 0.45 <= final_score <= 0.55:
        label = "neutral"
    else:
        label = "positive"

    return {
        "sentiment_label": label,
        "sentiment_score": round(final_score, 4),
        "model_used": feedback.user_type.capitalize()
    }
