import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "models/best_titanic_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return "Welcome to the Model Inference API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON request
        data = request.get_json(force=True)

        # Ensure that the features are provided in the correct format
        features = pd.DataFrame([data['features']])  # Wrap it in a list to make it a DataFrame

        # Ensure column names are present
        columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']  # Add more as needed
        features = features.reindex(columns=columns)  # Reindex the DataFrame

        # Make the prediction
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
