# Import necessary libraries
import pandas as pd
from pycaret.classification import setup

# Step 1: Data Collection and Preprocessing with PyCaret
def load_and_preprocess_data(url):
    """
    Load and preprocess the Titanic dataset using PyCaret.
    
    :param url: str, URL of the dataset
    :return: PyCaret setup object
    """
    # Load the dataset
    data = pd.read_csv(url)

    # Initial data overview
    print("Initial Data Overview:")
    print(data.head())

    # PyCaret Setup
    clf_setup = setup(data, 
                      target='Survived', 
                      ignore_features=['Cabin', 'Name', 'Ticket'], 
                      normalize=True,  # Data normalization
                      categorical_features=['Sex', 'Embarked'],  # Specify categorical features
                      session_id=42)  # Set a seed for reproducibility
    
    return clf_setup


def main():
    # Define the dataset URL
    dataset_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    
    # Load and preprocess the data
    clf_setup = load_and_preprocess_data(dataset_url)
    
    # Save the preprocessed data for model training
    print("Data Preprocessing Complete.")

if __name__ == "__main__":
    main()
