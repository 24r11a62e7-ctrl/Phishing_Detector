from utils.preprocess import clean_email
from utils.url_features import extract_url_features, FEATURE_NAMES

email = "URGENT! Verify your account at http://fake.com"
url = "http://paypa1-login.com"

print("Clean Email:")
print(clean_email(email))

print("\nURL Features:")
for name, value in zip(FEATURE_NAMES, extract_url_features(url)):
    print(f"{name:18s} {value}")
