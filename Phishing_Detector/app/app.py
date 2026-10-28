import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Make the project root importable when running "streamlit run app/app.py"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.preprocess import clean_email
from utils.url_features import extract_url_features, FEATURE_NAMES

st.set_page_config(page_title="Phishing Detection System", layout="wide")


@st.cache_resource
def load_models():
    email_model = joblib.load(os.path.join(ROOT, "models/email_model.pkl"))
    vectorizer = joblib.load(os.path.join(ROOT, "models/vectorizer.pkl"))
    url_model = joblib.load(os.path.join(ROOT, "models/url_model.pkl"))
    return email_model, vectorizer, url_model


email_model, vectorizer, url_model = load_models()

st.sidebar.title("Project Information")
st.sidebar.info(
    "Phishing Detection System\n\n"
    "Email model: TF-IDF + machine learning classifier\n\n"
    "URL model: 17 lexical URL features + machine learning classifier\n\n"
    "Technologies: Python, scikit-learn, Streamlit"
)

st.title("Phishing Detection System")
st.write("Check whether an email or a URL is likely to be a phishing attempt.")

tab1, tab2 = st.tabs(["Email Detection", "URL Detection"])

# ---------------- EMAIL ----------------
with tab1:
    st.subheader("Check Email")
    email_input = st.text_area("Enter Email Text", height=200)

    if st.button("Check Email"):
        if not email_input.strip():
            st.warning("Please enter the email text.")
        else:
            cleaned = clean_email(email_input)
            vector = vectorizer.transform([cleaned])
            prob = email_model.predict_proba(vector)[0][1]
            prediction = email_model.predict(vector)[0]

            c1, c2 = st.columns(2)
            c1.metric("Phishing probability", f"{prob * 100:.2f}%")
            c2.metric("Safe probability", f"{(1 - prob) * 100:.2f}%")
            st.progress(float(prob))

            if prediction == 1:
                st.error("Phishing Email Detected")
            else:
                st.success("Safe Email")

# ---------------- URL ----------------
with tab2:
    st.subheader("Check URL")
    url_input = st.text_input("Enter URL")

    if st.button("Check URL"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            features = extract_url_features(url_input)
            frame = pd.DataFrame([features], columns=FEATURE_NAMES)
            prob = url_model.predict_proba(frame)[0][1]
            prediction = url_model.predict(frame)[0]

            c1, c2 = st.columns(2)
            c1.metric("Phishing probability", f"{prob * 100:.2f}%")
            c2.metric("Safe probability", f"{(1 - prob) * 100:.2f}%")
            st.progress(float(prob))

            if prediction == 1:
                st.error("Phishing URL Detected")
            else:
                st.success("Safe URL")

            with st.expander("Extracted URL features"):
                st.dataframe(pd.DataFrame({"Feature": FEATURE_NAMES, "Value": features}),
                             hide_index=True)
