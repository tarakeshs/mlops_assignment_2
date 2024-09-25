import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pycaret.classification import load_model, setup, get_config

# Get the directory of the current file dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model dynamically
model_path = os.path.join(current_dir, '../models/best_titanic_model')

# Load the dataset to reinitialize PyCaret's environment
dataset_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(dataset_url)

# Initialize the PyCaret environment (re-run setup with the same configuration)
clf_setup = setup(data, 
                  target='Survived', 
                  ignore_features=['Cabin', 'Name', 'Ticket'], 
                  normalize=True,  # Normalize the features
                  categorical_features=['Sex', 'Embarked'],  # Treat 'Sex' and 'Embarked' as categorical
                  session_id=42)

# Load the trained model (which is a pipeline)
pipeline_model = load_model(model_path)
print("Transformation Pipeline and Model Successfully Loaded")

# Extract the actual model (RandomForestClassifier) from the pipeline
model = pipeline_model.steps[-1][1]

# Get the transformed test data (which is already encoded and preprocessed)
X_test = get_config('X_test')

# Apply the same transformations to ensure that the categorical variables are encoded as numeric values
X_test_transformed = pipeline_model[:-1].transform(X_test)

# Check the content of X_test_transformed to ensure it's numeric
print("Transformed X_test sample:\n", pd.DataFrame(X_test_transformed).head())

# Ensure the data is numeric for SHAP
if not pd.DataFrame(X_test_transformed).applymap(np.isreal).all().all():
    print("Non-numeric values detected in X_test_transformed!")

# Step 3: Create a SHAP explainer (use TreeExplainer for tree-based models)
explainer = shap.TreeExplainer(model)

# Step 4: Calculate SHAP values for the test dataset
shap_values = explainer.shap_values(X_test_transformed)

# Step 5: Visualize SHAP values
# Summary plot to show feature importance
print("Generating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", show=False)

# Save the SHAP summary plot
summary_plot_path = os.path.join(current_dir, "shap_summary_plot.png")
plt.savefig(summary_plot_path)
plt.show()
print(f"Summary plot saved at: {summary_plot_path}")

# Step 6: Force plot to explain a single prediction (first instance in the test set)
print("Generating SHAP Force Plot for a single instance...")
shap.initjs()

# Convert the transformed test set to a DataFrame for SHAP force plot
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=get_config('X_train').columns)

# Ensure that we're passing the right format for SHAP force plot
force_plot_path = os.path.join(current_dir, "shap_force_plot_instance_1.png")
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test_transformed_df.iloc[0], show=False)
plt.savefig(force_plot_path)
print(f"Force plot saved at: {force_plot_path}")
