import pandas as pd
from pycaret.classification import setup
from sklearn.impute import SimpleImputer
import numpy as np
from pandas_profiling import ProfileReport  # Optional: For AutoEDA
from pydantic_settings import BaseSettings


# Step 1: Data Collection and Preprocessing with PyCaret
def load_and_preprocess_data(url, autoeda=False):
    """
    Load and preprocess the Titanic dataset using PyCaret.
    
    :param url: str, URL of the dataset
    :param autoeda: bool, whether to perform AutoEDA using Pandas Profiling (optional)
    :return: PyCaret setup object
    """
    # Load the dataset
    data = pd.read_csv(url)

    # Initial data overview
    print("Initial Data Overview:")
    print(data.head())

    # Optional: Generate AutoEDA report using Pandas Profiling
    if autoeda:
        print("Generating AutoEDA report...")
        profile = ProfileReport(data, title="Titanic Dataset AutoEDA Report", explorative=True, config_file=None)
        profile.to_file("titanic_autoeda_report.html")
        print("AutoEDA report saved as 'titanic_autoeda_report.html'")

    # Step 2: Handling missing data
    # 'Age' and 'Embarked' columns have missing values
    # 'Age' will be imputed using the median as it is a continuous variable
    imputer_age = SimpleImputer(strategy='median')

    # 'Embarked' will be imputed using the most frequent value (mode) since it's a categorical variable
    imputer_embarked = SimpleImputer(strategy='most_frequent')

    # Apply the imputers to the relevant columns
    data['Age'] = imputer_age.fit_transform(data[['Age']])
    data['Embarked'] = imputer_embarked.fit_transform(data[['Embarked']]).ravel()

    # Step 3: Feature Engineering
    # Create a new feature 'FamilySize' as the sum of 'Parch' and 'SibSp' (number of family members)
    data['FamilySize'] = data['Parch'] + data['SibSp']

    # Convert 'Sex' column into numeric using map (0 = female, 1 = male)
    # This is necessary for the machine learning model to process the feature
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # Step 4: Drop irrelevant columns
    # 'PassengerId', 'Cabin', 'Name', and 'Ticket' are identifiers or features that don’t provide predictive power
    data = data.drop(columns=['PassengerId', 'Cabin', 'Name', 'Ticket'])

    # Step 5: PyCaret Setup
    # Use PyCaret to set up the classification model pipeline, with data normalization enabled
    clf_setup = setup(data, 
                      target='Survived',  # Target variable
                      ignore_features=['Cabin', 'Name', 'Ticket'],  # Ignore unnecessary features
                      normalize=True,  # Normalize the features for better model performance
                      categorical_features=['Embarked'],  # Treat 'Embarked' as a categorical feature
                      session_id=42)  # Set a seed for reproducibility
    
    return clf_setup

def main():
    # Define the dataset URL
    dataset_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    
    # Load and preprocess the data with optional AutoEDA (set autoeda=True to generate EDA report)
    clf_setup = load_and_preprocess_data(dataset_url, autoeda=True)
    
    # Save the preprocessed data for model training
    print("Data Preprocessing Complete.")
    

if __name__ == "__main__":
    main()