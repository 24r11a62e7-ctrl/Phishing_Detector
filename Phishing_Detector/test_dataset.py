import pandas as pd

emails = pd.read_csv("data/emails.csv")
urls = pd.read_csv("data/urls.csv")

print("Email Dataset:", emails.shape)
print(emails["label"].value_counts())
print(emails.head())

print("\nURL Dataset:", urls.shape)
print(urls["label"].value_counts())
print(urls.head())
