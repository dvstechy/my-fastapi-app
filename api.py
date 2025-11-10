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

# --- Request model ---
class FeedbackText(BaseModel):
    user_type: str  # "User" or "Farmer"

    # Shared fields
    text: str = ""
    overall_review: str = ""
    overall_rating: int = None

    # --- User feedback fields ---
    e_market_review: str = ""
    recipe_review: str = ""
    chatbot_review: str = ""
    contribution_review: str = ""
    e_market_rating: int = None
    recipe_rating: int = None
    chatbot_rating: int = None
    contribution_rating: int = None

    # --- Farmer feedback fields ---
    farmer_dashboard_review: str = ""
    farmer_products_review: str = ""
    farmer_orders_review: str = ""
    farmer_profile_review: str = ""
    farmer_dashboard_rating: int = None
    farmer_products_rating: int = None
    farmer_orders_rating: int = None
    farmer_profile_rating: int = None


# --- Load model based on user type ---
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


# --- Text cleaning helper ---
def clean_text(text: str):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


# --- Main prediction endpoint ---
@app.post("/predict-sentiment")
def predict_sentiment(feedback: FeedbackText):
    try:
        model, vectorizer = load_role_model(feedback.user_type)
    except Exception as e:
        return {"error": str(e)}

    # --- Combine reviews based on user type ---
    if feedback.user_type.lower() == "farmer":
        combined_text = " ".join([
            feedback.text,
            feedback.farmer_dashboard_review,
            feedback.farmer_products_review,
            feedback.farmer_orders_review,
            feedback.farmer_profile_review,
            feedback.overall_review,
        ])
        ratings = [
            feedback.farmer_dashboard_rating,
            feedback.farmer_products_rating,
            feedback.farmer_orders_rating,
            feedback.farmer_profile_rating,
            feedback.overall_rating,
        ]
    else:  # For normal User
        combined_text = " ".join([
            feedback.text,
            feedback.e_market_review,
            feedback.recipe_review,
            feedback.chatbot_review,
            feedback.contribution_review,
            feedback.overall_review,
        ])
        ratings = [
            feedback.e_market_rating,
            feedback.recipe_rating,
            feedback.chatbot_rating,
            feedback.contribution_rating,
            feedback.overall_rating,
        ]

    # --- Clean and vectorize ---
    cleaned_text = clean_text(combined_text)
    X = vectorizer.transform([cleaned_text])

    # --- Predict probabilities ---
    probs = model.predict_proba(X)[0]
    label = model.predict(X)[0]
    label_index = list(model.classes_).index(label)
    text_score = float(probs[label_index])  # confidence from text

    # --- Rating influence ---
    ratings = [r for r in ratings if r is not None]
    rating_score = ((np.mean(ratings) - 1) / 4) if ratings else 0

    # --- Weighted combination ---
    final_score = 0.7 * text_score + 0.3 * rating_score

    # --- Determine sentiment label ---
    if final_score < 0.45:
        label = "negative"
    elif 0.45 <= final_score <= 0.55:
        label = "neutral"
    else:
        label = "positive"

    return {
        "sentiment_label": label,
        "sentiment_score": round(final_score, 4),
        "model_used": feedback.user_type.capitalize(),
    }
