import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model from models folder
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/best_titanic_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return "Welcome to the Model Inference API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON request
    data = request.get_json(force=True)
    
    try:
        # Assuming the input is a list of feature values
        input_features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
