import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from utils.preprocess import clean_email

# Load dataset
df = pd.read_csv("data/emails.csv")
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# Clean the text
df["clean"] = df["email"].apply(clean_email)

# Split data (stratified, fixed seed)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
print("Train:", len(X_train), " Test:", len(X_test))

# Convert text into numbers - fitted on TRAINING emails only
vectorizer = TfidfVectorizer(stop_words="english", max_features=20000,
                             ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

# Train and compare models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(),
}

results = {}
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n{name}  accuracy: {results[name]['accuracy'] * 100:.2f}%")
    print(classification_report(y_test, pred, target_names=["Safe", "Phishing"], digits=4))

# Keep the model with the best F1-score for the app
best = max(results, key=lambda n: results[n]["f1"])
print("Best email model:", best)

joblib.dump(models[best], "models/email_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
json.dump({"best": best, "results": results, "train_size": len(X_train), "test_size": len(X_test)},
          open("models/email_metrics.json", "w"), indent=2)
print("Model saved successfully!")
