"""Prepare the two training datasets used by the Phishing Detection System.

Raw sources (placed in data/raw/):
  * Phishing_Email.csv - Kaggle "Phishing Email Detection" dataset (columns: Email Text, Email Type)
  * url_data.csv       - Malicious URL dataset from github.com/faizann24/
                         Using-machine-learning-to-detect-malicious-URLs (columns: url, label = good/bad)

Output (same column names the project already used):
  * data/emails.csv  -> email,label   (label 1 = phishing, 0 = safe)
  * data/urls.csv    -> url,label     (label 1 = phishing/malicious, 0 = safe)
"""
import pandas as pd

# ---------------- EMAILS ----------------
emails = pd.read_csv("data/raw/Phishing_Email.csv")
print("Raw email rows:", len(emails))

emails = emails.rename(columns={"Email Text": "email", "Email Type": "label"})
emails = emails[["email", "label"]]
emails = emails.dropna()                                   # 16 rows have no text
emails["email"] = emails["email"].astype(str)
emails = emails[emails["email"].str.strip().str.lower() != "empty"]
emails = emails.drop_duplicates(subset="email")            # avoid the same email in train and test
emails["label"] = emails["label"].map({"Safe Email": 0, "Phishing Email": 1})

emails.to_csv("data/emails.csv", index=False)
print("Clean email rows:", len(emails))
print(emails["label"].value_counts())

# ---------------- URLS ----------------
urls = pd.read_csv("data/raw/url_data.csv")
print("\nRaw URL rows:", len(urls))

urls = urls.dropna()
urls["url"] = urls["url"].astype(str).str.strip()
urls = urls.drop_duplicates(subset="url")
urls["label"] = urls["label"].map({"good": 0, "bad": 1})

urls.to_csv("data/urls.csv", index=False)
print("Clean URL rows:", len(urls))
print(urls["label"].value_counts())
