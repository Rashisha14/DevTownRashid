# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("boston_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features = np.array([input_features])
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)

