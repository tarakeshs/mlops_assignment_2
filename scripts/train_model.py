import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_and_preprocess_data():
    """
    Load Iris dataset and perform preprocessing: data cleaning, feature scaling, and splitting.
    :return: X_train, X_test, y_train, y_test: Train-test split data
    """
    # Load Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Rename the target column to 'target' for consistency
    df.rename(columns={'target': 'target'}, inplace=True)

    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # No scaling needed for Random Forest, but included for consistency
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_and_save_model(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning and save the best model.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Best trained model
    """
    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    print("Best Hyperparameters:", grid_search.best_params_)

    # Save the best model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved at {model_path}")

    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    :param model: Trained model
    :param X_test: Testing features
    :param y_test: Testing labels
    """
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train the model and save it
    best_model = train_and_save_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
