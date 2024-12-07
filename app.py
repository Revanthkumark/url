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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,Dense,GRU 
from tensorflow.keras.models import Sequential,save_model,load_model

from flask import Flask,request,render_template


# Initialize the Flask app
app = Flask(__name__,static_folder='static')

 
with open("tokenizer.pkl", "rb") as file:  # loading the tokenizier that which is pickle file to the tokenizer
        url_tokenizer = pickle.load(file)

output_data_label = {0: 'benign', 1: 'defacement', 2: 'malware', 3: 'phishing'}

vocab_size = len(url_tokenizer.word_index)+1

model = load_model('url_model_12.h5')
# this function is used for the prediction process and predict the output

def prediction_url(text)->str:
    sample = text.split(sep='/'or'//')
    pred_text = url_tokenizer.texts_to_sequences(sample)
    pred_text = [item for data in pred_text for item in data]
    print(pred_text)
    input_sequence = np.array(pred_text)
    input_sequence = np.expand_dims(input_sequence, axis=0)
    prediction = model.predict(input_sequence)# Here the prediction will be generated in the form of matraix having labels percentage
    return output_data_label[int(np.argmax(prediction))] # This is a dict that which will give the label out

#pred=prediction_url("wvmqquthmutualportmutual.000webhostapp.com")


# Route for the main page
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    result_class = None
    result_message = None
    result_icon = None
    #phishing_details = None
    if request.method == "POST":
        url_link = request.form.get("url", "")
        if url_link:  # Ensure input is not empty
            result = prediction_url(url_link)

            if result == 'benign':
                result_class = "benign"
                result_message = f"The {url_link} is a Good Website"
                result_icon = "✅"
                

            elif result == 'defacement':
                result_class = "defacement"
                result_message = f"The {url_link} is a Defacement Website"
                result_icon =  "⚠️"
                

            elif result == 'phishing':
                result_class = "phishing"
                result_message = f"The {url_link} is a Phishing Website  "
                result_icon =  "🚨"


            elif result == 'malware':
                result_class = 'malware'
                result_message = f"The {url_link} is a Malware Website"
                result_icon = "🛑"
                
    #print(f"Result Class: {result_class}")
    #print(f"Result Message: {result_message}")
    #print(f"Result Test:{phishing_details}")

    
    return render_template("index.html", 
                           result=result, 
                           result_class=result_class,
                           result_message = result_message,
                           result_icon = result_icon,
                           #phishing_details=phishing_details
                           )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)




