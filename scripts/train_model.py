import os
from pycaret.classification import setup, compare_models, tune_model, finalize_model, save_model, pull, get_config
import pandas as pd
import logging

# Setup logging to capture the experimentation process
logging.basicConfig(filename='experiment_logs.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Setup PyCaret environment and load the dataset
def setup_pycaret():
    """
    Load the Titanic dataset and set up the PyCaret environment.
    Logs and saves initial data overview for documentation.
    """
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)

    # Initial data overview
    logging.info("Initial Data Overview:\n%s", data.head())
    print("Initial Data Overview:")
    print(data.head())

    # Run PyCaret setup
    clf_setup = setup(data, 
                      target='Survived', 
                      ignore_features=['Cabin', 'Name', 'Ticket'], 
                      normalize=True,  # Data normalization
                      categorical_features=['Sex', 'Embarked'],  # Specify categorical features
                      session_id=42)  # Set a seed for reproducibility
    
    # Save the feature names for documentation
    features = get_config('X_train').columns.tolist()
    with open('feature_names.txt', 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")

    logging.info(f"Feature names saved: {features}")
    
    return clf_setup


# Step 2: Model Selection, Hyperparameter Tuning, and Finalization
def train_and_evaluate_model():
    """
    Train, tune, and evaluate the model using PyCaret.
    Logs the model selection and hyperparameter tuning process.
    Saves the comparison and tuning results for documentation.
    """
    # Call setup first
    logging.info("Setting up PyCaret environment.")
    setup_pycaret()

    # Step 2.1: Compare multiple models (sorted by Accuracy)
    logging.info("Comparing models based on Accuracy.")
    
    # The `compare_models()` function will train multiple models and return the top-performing one based on the chosen metric (Accuracy here).
    best_model = compare_models(sort='Accuracy')

    # Log the comparison of models
    logging.info("Best model: %s", best_model)
    
    # Save the comparison results to CSV for documentation
    comparison_df = pull()
    comparison_df.to_csv('model_comparison.csv')
    logging.info("Model comparison results saved to model_comparison.csv")

    # Step 2.2: Hyperparameter Tuning on the best model
    logging.info("Tuning hyperparameters for the best model.")
    
    tuned_model = tune_model(best_model)

    # Log tuned model details
    logging.info("Tuned model: %s", tuned_model)

    # Save the tuned model results to CSV for documentation
    tuned_results = pull()
    tuned_results.to_csv('tuned_model_results.csv')
    logging.info("Tuned model results saved to tuned_model_results.csv")

    # Step 2.3: Finalize the model (train on the entire dataset)
    logging.info("Finalizing the model after tuning.")
    
    final_model = finalize_model(tuned_model)

    # Create a directory for saving models
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the final model
    save_model(final_model, os.path.join(model_dir, 'best_titanic_model'))
    logging.info(f"Model saved to {model_dir}/best_titanic_model.pkl")

    print(f"Model saved to {model_dir}/best_titanic_model.pkl")

    # Log the entire model details
    logging.info(f"Final model details:\n{final_model}")


def main():
    """
    Main function to initiate the model training, tuning, and evaluation process.
    Logs all the steps and saves the models and results for documentation.
    """
    logging.info("Starting the model selection, training, and tuning process.")
    
    # Call the function to train and evaluate the model
    train_and_evaluate_model()

    logging.info("Model training process completed.")


if __name__ == "__main__":
    main()
