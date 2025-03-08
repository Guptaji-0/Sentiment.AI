import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = " ".join([word for word in word_tokenize(text) if word not in stopwords.words("english")])
    return text

def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.dropna(inplace=True)  # Remove missing values
    df["cleaned_text"] = df[df.columns[0]].apply(clean_text)  # Apply text cleaning
    return df
