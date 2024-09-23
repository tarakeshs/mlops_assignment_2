import os
from pycaret.classification import setup, compare_models, tune_model, finalize_model, predict_model, evaluate_model, save_model, get_config
import pandas as pd

# Step 1: Setup PyCaret environment and load the dataset
def setup_pycaret():
    """
    Load the Titanic dataset and set up the PyCaret environment.
    """
    # Load the dataset
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)

    # Initial data overview
    print("Initial Data Overview:")
    print(data.head())

    # Run PyCaret setup
    clf_setup = setup(data, 
                      target='Survived', 
                      ignore_features=['Cabin', 'Name', 'Ticket'], 
                      normalize=True,  # Data normalization
                      categorical_features=['Sex', 'Embarked'],  # Specify categorical features
                      session_id=42)  # Set a seed for reproducibility
    
    return clf_setup


# Step 2: Model Selection and Hyperparameter Tuning
def train_and_evaluate_model():
    """
    Train, tune, and evaluate the model using PyCaret.
    """
    # Call setup first
    setup_pycaret()

    # Step 2.1: Compare multiple models
    best_model = compare_models(sort='Accuracy')

    # Step 2.2: Hyperparameter Tuning
    tuned_model = tune_model(best_model)

    # Step 2.3: Finalize the model (retrain on entire dataset)
    final_model = finalize_model(tuned_model)

    # Save the final model under models folder
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    save_model(final_model, os.path.join(model_dir, 'best_titanic_model'))

    print(f"Model saved to {model_dir}/best_titanic_model.pkl")


def main():
    # Call the function to train and evaluate the model
    train_and_evaluate_model()

if __name__ == "__main__":
    main()
