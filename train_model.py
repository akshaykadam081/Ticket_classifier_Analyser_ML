# train_model.py

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("ticket_dataset.csv")

print(df.head())

# =========================
# REMOVE NULLS & DUPLICATES
# =========================

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["cleaned_text"] = df["ticket_text"].apply(clean_text)

# =========================
# FEATURE ENGINEERING
# =========================

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df["cleaned_text"])

# =========================
# LABEL ENCODING
# =========================

encoder = LabelEncoder()

y = encoder.fit_transform(df["category"])

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = LogisticRegression()

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\n=========================")
print("MODEL METRICS")
print("=========================")

print(f"Accuracy  : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall    : {recall}")
print(f"F1 Score  : {f1}")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n=========================")
print("CLASSIFICATION REPORT")
print("=========================")

print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================
# SAVE MODEL
# =========================

pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

pickle.dump(
    encoder,
    open("label_encoder.pkl", "wb")
)

print("\nModel saved successfully.")