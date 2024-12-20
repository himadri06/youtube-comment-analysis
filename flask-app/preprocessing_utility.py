# Define the preprocessing function
import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

def preprocess_comment(text):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        text = text.lower()

        # Remove trailing and leading whitespaces
        text = text.strip()

        # Remove newline characters
        text = re.sub(r'\n', ' ', text)

        # Remove non-alphanumeric characters, except punctuation
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        text = ' '.join([word for word in text.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

        return text
    except Exception as e:
        return text


