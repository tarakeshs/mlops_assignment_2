import logging
from data_preprocessing import load_and_preprocess_data  # Import preprocessing function
from train_model import train_and_evaluate_model  # Import model training function

# Setup logging
logging.basicConfig(filename='predict_and_train_logs.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def predict_and_train():
    """
    Run data preprocessing and model training.
    Logs the process for documentation and debugging.
    """
    logging.info("Starting data preprocessing and model training.")
    
    # Step 1: Run data preprocessing
    logging.info("Running data preprocessing.")
    dataset_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    load_and_preprocess_data(dataset_url)  # Preprocess the data
    
    # Step 2: Run model training
    logging.info("Running model training.")
    train_and_evaluate_model()  # Train and tune the model
    
    logging.info("Predict and Train process completed successfully.")
    print("Predict and Train process completed successfully.")

if __name__ == "__main__":
    predict_and_train()
