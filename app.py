from flask import Flask, request, jsonify, render_template
import joblib
from pre_processing import clean_text

# -------------------------------------------------
# CREATE FLASK APP
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# LOAD MODEL & VECTORIZER
# -------------------------------------------------
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("hate_speech_vectorizer.pkl")

# -------------------------------------------------
# STRONG HATE WORDS (RULE-BASED LAYER)
# -------------------------------------------------
STRONG_HATE_WORDS = [
    'nigger', 'bitch', 'fuck', 'faggot', 'retard', 'hoe',
    'stupid', 'kill', 'dumb', 'hate', 'worthless', 'trash',
    'idiot', 'ugly', 'die', 'moron', 'nigga', 'fucker'
]

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------------------------
# API: PREDICT HATE SPEECH (AJAX)
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("text", "")

    cleaned_text = clean_text(user_text)

    # ---------- RULE-BASED CHECK ----------
    if any(word in cleaned_text.split() for word in STRONG_HATE_WORDS):
        return jsonify({
            "prediction": "Hate Speech Detected",
            "confidence": 99
        })

    # ---------- ML MODEL CHECK ----------
    vector = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(vector)[0]

    hate_prob = probabilities[1] * 100
    not_hate_prob = probabilities[0] * 100

    if hate_prob >= not_hate_prob:
        prediction = "Hate Speech Detected"
        confidence = round(hate_prob, 2)
    else:
        prediction = "Not Hate Speech"
        confidence = round(not_hate_prob, 2)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

# -------------------------------------------------
# RUN SERVER (LOCAL)
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
