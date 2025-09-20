import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the pre-trained model and vectorizer
try:
    with open('fake_news_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please run `train_model.py` first.")
    st.stop()

# --- Text Preprocessing Function ---
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# --- Streamlit App Layout ---
st.title('Fake News Detector 🗞️')
st.markdown("### Enter a news article below to check its authenticity.")

# Text area for user input
user_input = st.text_area("Paste the news article here:", height=300, placeholder="Start typing or paste a news article here...")

# Button to trigger the prediction
if st.button('Predict'):
    if user_input:
        # Preprocess the user's input
        processed_input = word_drop(user_input)

        # Convert the processed text into features using the loaded vectorizer
        vectorized_input = vectorizer.transform([processed_input])

        # Make the prediction
        prediction = model.predict(vectorized_input)
        
        # Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.markdown("<p style='color: green; font-size: 24px;'>✅ Real News</p>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("<p style='color: red; font-size: 24px;'>❌ Fake News</p>", unsafe_allow_html=True)
            st.snow()
    else:
        st.warning("Please enter some text to make a prediction.")

st.markdown("---")
st.markdown("This application uses a Machine Learning model trained on a public dataset to classify news articles.")