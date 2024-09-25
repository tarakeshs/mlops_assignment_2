import pandas as pd
from pycaret.classification import setup
from sklearn.impute import SimpleImputer
import numpy as np

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

    # Step 2: Handling missing data
    # 'Age' and 'Embarked' columns have missing values, we'll use median and mode for imputation
    imputer_age = SimpleImputer(strategy='median')
    imputer_embarked = SimpleImputer(strategy='most_frequent')

    data['Age'] = imputer_age.fit_transform(data[['Age']])
    data['Embarked'] = imputer_embarked.fit_transform(data[['Embarked']]).ravel()

    # Step 3: Feature Engineering
    # Create a new feature: FamilySize = Parch + SibSp
    data['FamilySize'] = data['Parch'] + data['SibSp']

    # Convert 'Sex' column into numeric using map (0 = female, 1 = male)
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # Step 4: PyCaret Setup
    clf_setup = setup(data, 
                      target='Survived', 
                      ignore_features=['Cabin', 'Name', 'Ticket'], 
                      normalize=True,  # Data normalization
                      categorical_features=['Embarked'],  # Specify categorical features
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
