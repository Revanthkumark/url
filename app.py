import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import LabelEncoder
import tensorflow
import keras
import flask
import pickle
import logging

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential, save_model, load_model
from flask import Flask, request, render_template

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__, static_folder='static')

# Load tokenizer
try:
    with open("tokenizer.pkl", "rb") as file:  # Loading the tokenizer that is a pickle file
        url_tokenizer = pickle.load(file)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    raise

output_data_label = {0: 'benign', 1: 'defacement', 2: 'malware', 3: 'phishing'}
vocab_size = len(url_tokenizer.word_index) + 1

# Load the model
try:
    model = load_model('url_model_12.h5')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# This function is used for the prediction process and predicts the output
def prediction_url(text) -> str:
    try:
        # Tokenizing the URL and making prediction
        sample = text.split('/')  # Simple split by '/'
        pred_text = url_tokenizer.texts_to_sequences(sample)
        pred_text = [item for data in pred_text for item in data]
        logger.debug(f"Tokenized input: {pred_text}")
        
        input_sequence = np.array(pred_text)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        
        prediction = model.predict(input_sequence)
        logger.debug(f"Model prediction: {prediction}")
        
        # Return the predicted label
        return output_data_label[int(np.argmax(prediction))]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return "Error in prediction"

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    result_class = None
    result_message = None
    result_icon = None

    if request.method == "POST":
        url_link = request.form.get("url", "")
        result = prediction_url(url_link)

        if result == 'benign':
            result_class = "benign"
            result_message = f"The {url_link} is a Good Website"
            result_icon = "✅"

        elif result == 'defacement':
            result_class = "defacement"
            result_message = f"The {url_link} is a Defacement Website"
            result_icon = "⚠️"

        elif result == 'phishing':
            result_class = "phishing"
            result_message = f"The {url_link} is a Phishing Website"
            result_icon = "🚨"

        elif result == 'malware':
            result_class = 'malware'
            result_message = f"The {url_link} is a Malware Website"
            result_icon = "🛑"
    
    return render_template("index.html", 
                           result=result, 
                           result_class=result_class,
                           result_message=result_message,
                           result_icon=result_icon)

# Run the Flask app
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error running Flask app: {e}")
