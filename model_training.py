import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("labeled_data.csv")
data = data[['tweet', 'class']]         
data.columns = ['text', 'label']       

# labels
data['label'] = data['label'].apply(lambda x: 1 if x == 0 else 0)

# Clean text
def clean_text(text):
    text = text.lower()                  
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"@\w+", "", text)     
    text = re.sub(r"[^a-z\s]", "", text) 
    return text

data['cleaned_text'] = data['text'].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "hate_speech_model.pkl")
joblib.dump(vectorizer, "hate_speech_vectorizer.pkl")
print("\nModel and vectorizer saved")
