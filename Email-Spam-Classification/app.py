from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('spam_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

def preprocess_email(text, tokenizer, maxlen=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen)
    return padded

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['email_text']
    padded_input = preprocess_email(text, tokenizer)
    prediction = model.predict(padded_input)[0][0]
    label = 'Spam' if prediction > 0.5 else 'Ham'
    return render_template('result.html', result=label)

if __name__ == '__main__':
    app.run(debug=True)