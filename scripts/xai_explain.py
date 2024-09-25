import os
import shap
import joblib
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename='xai_logs.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model():
    """
    Load the trained model from the saved file.
    """
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models/best_titanic_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model

def load_data():
    """
    Load the Titanic dataset and preprocess it to match the model's expected features.
    """
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)
    
    # Preprocess the dataset to match the training phase
    data['FamilySize'] = data['Parch'] + data['SibSp']
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # Fill missing values in 'Embarked' with the most common value (mode)
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Include 'PassengerId', 'Embarked' and all necessary features
    features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Embarked']
    
    # Drop rows with missing values
    data = data[features].dropna()

    logging.info(f"Data loaded and preprocessed for SHAP analysis")
    return data

def preprocess_data_for_model(model, data):
    """
    Pass the data through the model's preprocessing pipeline.
    This ensures the data matches the features used during training.
    """
    logging.info("Preprocessing data through the model's pipeline.")
    
    # Apply the model's preprocessing pipeline
    Xt, _ = model._memory_full_transform(model, data, None, with_final=False)
    
    logging.info(f"Data preprocessed: {Xt.shape}")
    return Xt

def model_predict_proba(model, X):
    """
    Wrapper function for the model's predict_proba method.
    Ensures the input is in the correct format.
    """
    return model.predict_proba(X)

def explain_model(model, data):
    """
    Apply SHAP to explain model predictions.
    """
    logging.info("Applying SHAP to explain model predictions.")
    
    # Preprocess the data using the model's pipeline
    data_preprocessed = preprocess_data_for_model(model, data)
    
    # Convert preprocessed data to NumPy array for SHAP
    X_np = data_preprocessed.values

    # Create a SHAP explainer using the model's predict_proba function
    explainer = shap.KernelExplainer(lambda x: model_predict_proba(model, x), X_np)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_np)
    
    # Visualize the SHAP values for the first prediction (for the positive class)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], data.iloc[0, :], show=True)

    # Summary plot for feature importance (positive class)
    shap.summary_plot(shap_values[1], X_np)

    logging.info("SHAP analysis completed and visualized.")

def main():
    """
    Main function to load the model, data, and apply SHAP.
    """
    logging.info("Starting XAI (SHAP) analysis.")
    
    # Load the trained model
    model = load_trained_model()
    
    # Load the dataset
    data = load_data()
    
    # Apply SHAP to explain model predictions
    explain_model(model, data)
    
    logging.info("XAI (SHAP) analysis completed.")

if __name__ == "__main__":
    main()
