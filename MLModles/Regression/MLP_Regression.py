import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import random
from datetime import datetime

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

# Read the data
data = pd.read_excel(file_path)


# Data Preprocessing function
def preprocess_data(data):
    # Extract features and target variable
    X = data.drop(['Mol_ID', 'Q_band'], axis=1) # Target variable changed to Q_band
    y = data['Q_band'] # Target variable changed to Q_band
    mol_ids = data['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)

    # Get the names of the retained features
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]

    # Convert back to DataFrame
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    # Print feature count changes
    print(f"Original number of features: {X.shape[1]}")
    print(f"Filtered number of features: {X_filtered_df.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features


# MLP Hyperparameter Search function
def optimize_mlp(X_train, y_train, X_test, y_test, n_iter=200):
    print("\n=== MLP Model Hyperparameter Optimization ===")

    # Define MLP parameter search space
    param_dist = {
        'hidden_layer_sizes': [
            (32,), (64,), (128,),  # Single hidden layer
            (32, 16), (64, 32), (128, 64),  # Double hidden layer
            (64, 32, 16), (128, 64, 32)  # Triple hidden layer
        ],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': uniform(0.0001, 0.1),  # L2 regularization coefficient
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': uniform(0.001, 0.1),  # Initial learning rate
        'max_iter': randint(100, 1000),  # Maximum number of iterations
        'batch_size': ['auto'] + list(randint(16, 128).rvs(5)),  # Batch size
        'momentum': uniform(0.8, 0.2)  # Momentum (used by sgd solver)
    }

    # Initialize MLP Regressor
    mlp = MLPRegressor(random_state=SEED, early_stopping=True, validation_fraction=0.2)

    # Randomized Search
    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='r2',
        cv=5,  # 5-fold cross-validation
        random_state=SEED,
        n_jobs=-1,  # Use all available CPUs
        verbose=1
    )

    # Execute search
    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)

    # Print best parameters
    print("\n=== Best Parameters ===")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")

    # Predict using the best model
    best_mlp = random_search.best_estimator_
    y_train_pred = best_mlp.predict(X_train)
    y_test_pred = best_mlp.predict(X_test)

    # Evaluation
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nBest Model Training Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Best Model Testing Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

    # Print training iteration info
    print(f"\nTraining iterations: {best_mlp.n_iter_}")
    if best_mlp.loss_curve_ is not None and len(best_mlp.loss_curve_) > 0:
        print(f"Final training loss: {best_mlp.loss_curve_[-1]:.6f}")

    return y_test_pred, test_mse, test_r2, random_search.best_params_


# Data Preprocessing
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED
)

print(f"Number of samples in training set: {X_train.shape[0]}")
print(f"Number of samples in testing set: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Execute MLP hyperparameter search
mlp_pred, mlp_mse, mlp_r2, best_mlp_params = optimize_mlp(
    X_train, y_train, X_test, y_test, n_iter=200
)

print(f"\nFinal Testing Set R² for the Best Model: {mlp_r2:.4f}")


# 10-Fold Cross-Validation
print("\n=== 10-Fold Cross-Validation Evaluation ===")
# Initialize 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

# Store evaluation metrics for each fold
cv_mse_scores = []
cv_r2_scores = []

# Get the best model parameters
best_params = best_mlp_params

# Perform 10-fold cross-validation on the training set
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    # Split the current fold into training and validation sets
    x_cv_train, x_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Create model using the best parameters found
    cv_model = MLPRegressor(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.2,
        **best_params
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