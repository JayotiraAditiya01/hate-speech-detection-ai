import joblib
from pre_processing import clean_text

model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("hate_speech_vectorizer.pkl")

strong_hate_words = [
    'nigger', 'bitch', 'fuck', 'faggot', 'retard', 'hoe',
    'stupid', 'kill', 'dumb', 'hate', 'worthless', 'trash',
    'idiot', 'ugly', 'die', 'moron', 'nigga'
]

user_input = input("Enter a sentence to check for hate speech: ")
cleaned = clean_text(user_input)

if any(word in cleaned.split() for word in strong_hate_words):
    print("HATE SPEECH DETECTED")
else:
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    if prediction[0] == 1:
        print("HATE SPEECH DETECTED")
    else:
        print("NOT HATE SPEECH")
