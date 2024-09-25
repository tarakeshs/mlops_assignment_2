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

        # Create a DataFrame from the input features
        features = pd.DataFrame([data['features']])

        # Add a dummy PassengerId to the DataFrame as it's required by the model
        features['PassengerId'] = 0 

        # Ensure all required columns are present
        columns = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
        
        # Reindex the DataFrame to ensure correct column ordering
        features = features.reindex(columns=columns)

        # Make the prediction
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
