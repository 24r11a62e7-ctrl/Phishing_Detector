import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from utils.url_features import extract_url_features, FEATURE_NAMES

# Load dataset
df = pd.read_csv("data/urls.csv")
df = df.dropna()
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# Convert URLs into features
X = pd.DataFrame([extract_url_features(u) for u in df["url"]], columns=FEATURE_NAMES)
y = df["label"].values

# Split data (stratified, fixed seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", len(X_train), " Test:", len(X_test))

# Train and compare models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=3,
                                            random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n{name}  accuracy: {results[name]['accuracy'] * 100:.2f}%")
    print(classification_report(y_test, pred, target_names=["Safe", "Phishing"], digits=4))

best = max(results, key=lambda n: results[n]["f1"])
print("Best URL model:", best)

joblib.dump(models[best], "models/url_model.pkl", compress=3)
json.dump({"best": best, "results": results, "train_size": len(X_train), "test_size": len(X_test),
           "feature_names": FEATURE_NAMES},
          open("models/url_metrics.json", "w"), indent=2)
print("URL model saved successfully!")
