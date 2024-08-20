import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# File paths for saving and loading
MODEL_PATH = 'model.joblib'
VECTOR_PATH = 'vectorizer.joblib'
ENCODER_PATH = 'label_encoder.joblib'

# Function to train and save the model
def train_and_save_model():
    # Load the data
    train_data = pd.read_csv('train_data.txt', delimiter=':::', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')

    # Preprocess the text data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(train_data['DESCRIPTION'])

    # Encode the target labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['GENRE'])

    # Train a classifier
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_tfidf, y_train)

    # Save the model and other objects
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTOR_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.2f}")

# Function to load the model and other objects
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTOR_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, vectorizer, label_encoder

# Train and save the model if not already done
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load the model and vectorizer
model, vectorizer, label_encoder = load_model_and_vectorizer()

# Function to predict genre for a new description
def predict_genre(description):
    description_tfidf = vectorizer.transform([description])
    predicted_genre = model.predict(description_tfidf)
    return label_encoder.inverse_transform(predicted_genre)[0]

# Interactive input loop
while True:
    user_input = input("Enter your movie description or type exit to quit:")
    if user_input.lower() == 'exit':
        break
    predicted_genre = predict_genre(user_input)
    print(f"Predicted Genre: {predicted_genre}")
    