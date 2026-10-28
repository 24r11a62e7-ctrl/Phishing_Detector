import re


def clean_email(text):
    """Normalise email text before TF-IDF vectorisation."""
    if not isinstance(text, str):
        text = ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)   # keep the fact that a link was present
    text = re.sub(r"\S+@\S+", " emailtoken ", text)          # e-mail addresses
    text = re.sub(r"\d+", " ", text)                          # numbers
    text = re.sub(r"[^\w\s]", " ", text)                      # punctuation
    text = re.sub(r"\s+", " ", text).strip()                  # extra spaces

    return text
