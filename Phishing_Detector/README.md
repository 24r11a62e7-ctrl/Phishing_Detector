# Phishing Detection System

A machine learning project that detects **phishing emails** and **phishing / malicious URLs**,
served through an interactive Streamlit web application.

## Datasets
| File | Source | Rows (after cleaning) |
|---|---|---|
| `data/emails.csv` | Kaggle – *Phishing Email Detection* (subhajournal) | 17,536 (10,979 safe, 6,557 phishing) |
| `data/urls.csv` | Malicious URL dataset – github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs | 411,244 (344,800 safe, 66,444 malicious) |

`prepare_data.py` converts the raw downloads in `data/raw/` into these files
(drops empty rows and duplicates, maps labels to 0 = safe, 1 = phishing).

## Models
* **Email:** text cleaning (`utils/preprocess.py`) → TF-IDF (unigrams + bigrams, 20,000 terms, fitted on
  training emails only) → Random Forest, Logistic Regression and Multinomial Naive Bayes are compared;
  the best F1-score model is saved.
* **URL:** 17 lexical features (`utils/url_features.py`) → Random Forest, Decision Tree and Logistic
  Regression are compared; the best F1-score model is saved.

Both use a stratified 80:20 train/test split with `random_state=42`.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Email – Logistic Regression (saved) | 97.83% | 0.984 | 0.958 | 0.971 |
| URL – Random Forest (saved) | 92.97% | 0.875 | 0.659 | 0.752 |

## Run
```bash
pip install -r requirements.txt
python prepare_data.py      # only if you start from data/raw/
python train_email.py
python train_url.py
streamlit run app/app.py
```
