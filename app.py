import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string


# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Set NLTK data directory (if needed)
nltk_data_dir = 'd:\\apps\\Anaconda\\envs\\myenv\\nltk_data'
nltk.data.path.append(nltk_data_dir)

# Define the tokenizer function
def tokenizer(txt):
    stemmer = nltk.stem.SnowballStemmer('english')
    txt = ''.join([char for char in txt if char not in string.punctuation])
    return [stemmer.stem(token) for token in nltk.word_tokenize(txt.lower())]

# Load the model and vectorizer
try:
    vectorizer = joblib.load('vectorizer.pkl')  # Ensure the correct filename
    model = joblib.load('model.pkl')  # Ensure the correct filename
    st.success("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Load the dataset
try:
    data = pd.read_csv('IMDB Dataset.csv')  # Ensure the correct filename
    st.success("Dataset loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the dataset: {e}")
    st.stop()

# Streamlit UI
st.title("🎬 IMDB Review Analysis App")

# User input section
st.sidebar.header("User   Input")
review_text = st.sidebar.text_area("Enter your IMDB review here:", "")

if st.sidebar.button("Analyze Review"):
    if review_text:
        try:
            input_vector = vectorizer.transform([review_text])
            prediction = model.predict(input_vector)
            
            st.subheader("Prediction:")
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.markdown(f"<h3 style='color: #2FF3E0;'>{sentiment}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter a review to analyze.")

# Data insights section
st.header("Data Insights")

if 'sentiment' in data.columns:
    sentiment_counts = data['sentiment'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette='viridis')
    ax.set_title("Distribution of Review Sentiments")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.error("Sentiment column not found in the dataset.")

