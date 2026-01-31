from flask import Flask, request, jsonify, render_template
import joblib
from pre_processing import clean_text

# -------------------------------------------------
# CREATE FLASK APP
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# LOAD MODEL & VECTORIZER (LOAD ONCE)
# -------------------------------------------------
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("hate_speech_vectorizer.pkl")

# -------------------------------------------------
# STRONG HATE WORDS (KEEPING + EXTENSIBLE)
# -------------------------------------------------
strong_hate_words = [
    'nigger', 'bitch', 'fuck', 'faggot', 'retard', 'hoe',
    'stupid', 'kill', 'dumb', 'hate', 'worthless', 'trash',
    'idiot', 'ugly', 'die', 'moron', 'nigga', 'fucker'
]

# -------------------------------------------------
# HOME PAGE (FRONTEND)
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------------------------
# API PREDICTION (AJAX + CONFIDENCE)
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text.strip():
        return jsonify({
            "prediction": "Invalid Input",
            "confidence": 0
        })

    cleaned_text = clean_text(user_text)

    # ---------- RULE-BASED CHECK ----------
    if any(word in cleaned_text.split() for word in strong_hate_words):
        prediction = "Hate Speech"
        confidence = 99.0

    # ---------- ML MODEL CHECK ----------
    else:
        vector = vectorizer.transform([cleaned_text])
        probabilities = model.predict_proba(vector)[0]

        hate_prob = probabilities[1] * 100
        not_hate_prob = probabilities[0] * 100

        if hate_prob >= not_hate_prob:
            prediction = "Hate Speech"
            confidence = round(hate_prob, 2)
        else:
            prediction = "Not Hate Speech"
            confidence = round(not_hate_prob, 2)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
