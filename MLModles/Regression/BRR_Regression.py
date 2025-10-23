import os
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold # Import KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import BayesianRidge  # Bayesian Ridge Regression (suitable for continuous value prediction)
from scipy.stats import uniform

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Construct absolute path for data loading
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined
    current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'merged_data.xlsx')
print(f'Attempting to load data file from path: {file_path}')

# Check file existence
if not os.path.exists(file_path):
    # Fallback to direct filename if absolute path fails, then raise if not found
    file_path = 'merged_data.xlsx'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Data file not found, please check path: merged_data.xlsx')

# Read the merged data
data = pd.read_excel(file_path)


# Data Preprocessing (consistent with previous)
def preprocess_data(data):
    # Extract molecule ID and target variable
    X = data.drop(['Mol_ID', 'Q_band'], axis=1) # Target variable changed to Q_band
    y = data['Q_band'] # Target variable changed to Q_band
    mol_ids = data['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)

    # Get the names of the retained features
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]

    # Convert back to DataFrame and remove missing values
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)
    X_filtered_df = X_filtered_df.dropna(axis=1)

    # Print feature count change
    print(f"Original number of features: {X.shape[1]}")
    print(f"Filtered number of features: {X_filtered_df.shape[1]}")

    # Power transformation to make features closer to Gaussian distribution
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    X_transformed = transformer.fit_transform(X_filtered_df)
    X_transformed_df = pd.DataFrame(X_transformed, columns=X_filtered_df.columns)

    return X_transformed_df, y, mol_ids, transformer, filtered_features


# Bayesian Ridge Regression Hyperparameter Search function
def optimize_bayesian_ridge(X_train, y_train, X_test, y_test, n_iter=50):
    print("\n=== Bayesian Ridge Regression Model Hyperparameter Optimization ===")

    # Initialize Bayesian Ridge Regression model
    br = BayesianRidge()

    # Define parameter search space (parameters suitable for regression tasks)
    param_dist = {
        'alpha_1': uniform(1e-8, 1e-4),  # Shape parameter for prior distribution alpha
        'alpha_2': uniform(1e-8, 1e-4),
        'lambda_1': uniform(1e-8, 1e-4),  # Shape parameter for prior distribution lambda
        'lambda_2': uniform(1e-8, 1e-4),
        'normalize': [True, False]  # Whether to standardize features (optional here as pre-processed)
    }

    # Randomized Search
    random_search = RandomizedSearchCV(
        estimator=br,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )

    # Execute search
    print("Starting Bayesian Ridge hyperparameter search...")
    random_search.fit(X_train, y_train)

    # Print best parameters
    print("\n=== Best Parameters ===")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")

    # Predict using the best model
    best_br = random_search.best_estimator_
    y_train_pred = best_br.predict(X_train)
    y_test_pred = best_br.predict(X_test)

    # Evaluate
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nBest Model Training Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Best Model Testing Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

    return best_br, y_test_pred, test_mse, test_r2, random_search.best_params_


# Data Preprocessing
X_processed, y, mol_ids, transformer, filtered_features = preprocess_data(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED
)

print(f"Number of samples in training set: {X_train.shape[0]}")
print(f"Number of samples in testing set: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Execute Bayesian Ridge hyperparameter search
best_br_model, br_pred, br_mse, br_r2, best_br_params = optimize_bayesian_ridge(
    X_train, y_train, X_test, y_test, n_iter=50
)

print(f"\nFinal Testing Set R² for the Best Model: {br_r2:.4f}")


# 10-Fold Cross-Validation
print("\n=== 10-Fold Cross-Validation Evaluation ===")
# Initialize 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

# Store evaluation metrics for each fold
cv_mse_scores = []
cv_r2_scores = []

# Perform 10-fold cross-validation on the training set
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    # Split the current fold into training and validation sets
    x_cv_train, x_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Create model using the best parameters found in the optimization step
    cv_model = BayesianRidge(
        **best_br_params
    )

    # Train the model
    cv_model.fit(x_cv_train, y_cv_train)

    # Validation set prediction
    y_cv_pred = cv_model.predict(x_cv_val)

    # Calculate evaluation metrics
    cv_mse = mean_squared_error(y_cv_val, y_cv_pred)
    cv_r2 = r2_score(y_cv_val, y_cv_pred)

    # Store metrics
    cv_mse_scores.append(cv_mse)
    cv_r2_scores.append(cv_r2)

    # Print results for the current fold
    print(f"Fold {fold}: MSE = {cv_mse:.4f}, R² = {cv_r2:.4f}")

# Calculate and print the average cross-validation results
print("\n10-Fold Cross-Validation Average Results:")
print(f"Average MSE: {np.mean(cv_mse_scores):.4f} (±{np.std(cv_mse_scores):.4f})")
print(f"Average R²: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")