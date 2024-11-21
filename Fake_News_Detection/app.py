
import streamlit as st
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Load pre-trained model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    st.error("Model and vectorizer files are missing.")

def predict_news(news_text):
    """Predict whether the given news is fake or real."""
    vectorized_text = vectorizer.transform([news_text])
    prediction = model.predict(vectorized_text)
    return "Fake News" if prediction == 0 else "Real News"

st.title("Fake News Detection")

# Input: User can type or upload an image
news_input_type = st.radio("Choose input type:", ("Text", "Image"))

if news_input_type == "Text":
    user_input = st.text_area("Enter the news text here:")
    if st.button("Check News"):
        if user_input:
            result = predict_news(user_input)
            st.success(f"The news is: {result}")
        else:
            st.warning("Please enter some news text.")
elif news_input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image of the news text:", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        extracted_text = pytesseract.image_to_string(image)
        st.text_area("Extracted Text:", extracted_text, disabled=True)
        if st.button("Check News"):
            if extracted_text.strip():
                result = predict_news(extracted_text)
                st.success(f"The news is: {result}")
            else:
                st.warning("No text found in the image.")
    