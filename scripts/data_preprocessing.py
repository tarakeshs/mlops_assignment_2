import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def load_and_preprocess_data():
    """
    Load Iris dataset and perform preprocessing: data cleaning, feature scaling, and splitting.
    :return: X_train, X_test, y_train, y_test: Train-test split data
    """
    # Step 1: Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target  # Add the target column

    # Step 2: Data cleaning - No missing values in Iris dataset
    # If missing values were present, we would handle them here

    # Step 3: Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Step 4: Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Print shapes of the resulting datasets
    print("Training set size:", X_train.shape, y_train.shape)
    print("Testing set size:", X_test.shape, y_test.shape)
