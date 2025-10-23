import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# Removed: Font settings for Chinese display.

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Read the merged data
data = pd.read_excel('merged_data.xlsx')

# Data Preprocessing function
def preprocess_data(data):
    # Extract molecule ID and target variable
    X = data.drop(['Mol_ID', 'Q_band'], axis=1) # Q_hand changed to Q_band
    y = data['Q_band'] # Q_hand changed to Q_band
    mol_ids = data['Mol_ID']

    # Check and remove columns with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)

    # Get the names of the retained features
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]

    # Convert the filtered data back to a DataFrame
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    # Print the number of features before and after filtering
    print(f"Original number of features: {X.shape[1]}")
    print(f"Filtered number of features: {X_filtered_df.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features

# Call the preprocessing function
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=SEED)

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of testing samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Train GBDT model directly using best parameters
print("\n=== Training GBDT Model with Best Parameters ===")

# Best parameters
best_params = {
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'subsample': 0.9
}

print("Using parameters:", best_params)

# Create and train the model
best_gbdt_model = GradientBoostingRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    subsample=best_params['subsample'],
    random_state=SEED
)

# Train the model
best_gbdt_model.fit(X_train, y_train)

# Predict and evaluate
y_test_pred = best_gbdt_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Training set evaluation
y_train_pred = best_gbdt_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTraining Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Testing Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# SHAPAnalysis Analysis
print("\n=== Starting SHAPAnalysis Analysis ===")

# Perform SHAPAnalysis analysis using the best model
# Use TreeExplainer for GBDT model
explainer = shap.TreeExplainer(best_gbdt_model)
shap_values = explainer.shap_values(X_processed)

# Calculate feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = X_processed.columns

# Get the top 10 important features
important_indices = np.argsort(feature_importance)[-10:][::-1]
important_features = feature_names[important_indices].tolist()
important_values = feature_importance[important_indices].tolist()

# Print the top 10 important features
print("\nTop 10 Important Features:")
for i, (feature, importance) in enumerate(zip(important_features, important_values)):
    print(f"{i+1}. {feature}: {importance:.6f}")

# Create SHAPAnalysis summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_processed, max_display=10, show=False)
plt.title('GBDT SHAPAnalysis analysis', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('gbdt_shap_analysis.png', dpi=1200, bbox_inches='tight')
plt.close()


print("\n=== SHAPAnalysis Analysis Complete ===")