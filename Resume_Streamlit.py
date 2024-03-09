import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the clean_resume_text function
def clean_resume_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove RT (Retweet)
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)

    # Remove Hashtags and Mentions
    text = re.sub(r'#\w+|\@\w+', '', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Load the data
data = pd.read_csv('UpdatedResumeDataSet.csv')

# Clean the 'Resume' column
data['Cleaned_Resume'] = data['Resume'].apply(clean_resume_text)

# Encode the 'Category' column
label_encoder = LabelEncoder()
data['Category_Num'] = label_encoder.fit_transform(data['Category'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(data['Cleaned_Resume'])

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_tfidf, data['Category_Num'])

# Streamlit app
st.title("Resume Category Prediction")

# Inform the user about the categories that can be predicted
categories = data['Category'].unique()
categories_table = [categories[i:i+5] for i in range(0, len(categories), 5)]

# Add line breaks
st.markdown("<br>", unsafe_allow_html=True)

st.write("The following categories can be predicted correctly:")
table_data = []
for row in categories_table:
    # Pad with empty strings if the row has fewer than 5 categories
    padded_row = row + [''] * (5 - len(row)) if len(row) < 5 else row
    table_data.append(padded_row)

# Display the table without row or column headings
table_html = "<table><tbody>"
for row in table_data:
    table_html += "<tr>"
    for cell in row:
        table_html += f"<td>{cell}</td>"
    table_html += "</tr>"
table_html += "</tbody></table>"

st.write(table_html, unsafe_allow_html=True)

# Add line breaks
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Input field for user's resume
user_resume = st.text_area("ENTER YOUR RESUME:")

# Add line breaks
st.markdown("<br>", unsafe_allow_html=True)

# Predict the category
if st.button("PREDICT"):
    cleaned_resume = clean_resume_text(user_resume)
    tfidf_resume = tfidf_vectorizer.transform([cleaned_resume])
    category_num = nb_classifier.predict(tfidf_resume)[0]
    category = label_encoder.inverse_transform([category_num])[0]

    # Add line breaks
    st.markdown("<br>", unsafe_allow_html=True)

    st.write(f"PREDICTED CATEGORY: {category}")