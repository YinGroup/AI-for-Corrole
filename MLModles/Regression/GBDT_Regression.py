import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
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

# Read the merged data
data = pd.read_excel(file_path)

# Data Preprocessing function
def preprocess_data(data):
    # Extract molecule ID and target variable
    X = data.drop(['Mol_ID', 'Q_band'], axis=1) # Target variable changed to Q_band
    y = data['Q_band'] # Target variable changed to Q_band
    mol_ids = data['Mol_ID']

    # Check and remove columns with zero variance (features with variance 0)
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

print(f"Number of samples in training set: {X_train.shape[0]}")
print(f"Number of samples in testing set: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Define manual parameter search function (Random Search equivalent)
def optimize_gbdt(X_train, y_train, X_test, y_test, n_combinations=500):
    print("\n=== GBDT Model Hyperparameter Optimization ===")

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # Calculate the total number of possible combinations
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"Total number of possible parameter combinations: {total_combinations}")

    # Adjust the requested number if the total is smaller
    n_combinations = min(n_combinations, total_combinations)
    print(f"Will test {n_combinations} random parameter combinations")

    # Generate random parameter combinations
    param_combinations = []

    # First, add some baseline combinations
    base_combinations = [
        # Base combo 1: Low learning rate, medium depth
        {
            'n_estimators': 200,
            'learning_rate': 0.01,
            'max_depth': 5,
            'min_samples_split': 3,
            'min_samples_leaf': 2,
            'subsample': 0.8
        },
        # Base combo 2: Medium learning rate, shallow tree
        {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8
        },
        # Base combo 3: Medium learning rate, deep tree
        {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8
        },
        # Base combo 4: High learning rate, medium depth
        {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 3,
            'min_samples_leaf': 2,
            'subsample': 0.7
        },
        # Base combo 5: More estimators, low learning rate
        {
            'n_estimators': 300,
            'learning_rate': 0.01,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8
        }
    ]
    param_combinations.extend(base_combinations)

    # Then randomly generate the remaining combinations
    while len(param_combinations) < n_combinations:
        params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'max_depth': random.choice(param_grid['max_depth']),
            'min_samples_split': random.choice(param_grid['min_samples_split']),
            'min_samples_leaf': random.choice(param_grid['min_samples_leaf']),
            'subsample': random.choice(param_grid['subsample'])
        }

        # Ensure no duplicate parameter combinations
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []

    # Create timestamp for saving intermediate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}")
        print(params)

        # Create and train the model
        model = GradientBoostingRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            subsample=params['subsample'],
            random_state=SEED
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate on the test set
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Evaluate on the training set
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Store results
        results.append({
            'params': params,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_r2': test_r2
        })

        print(f"Training Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Testing Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

        # Update the best model
        if test_r2 > best_score:
            best_score = test_r2
            best_params = params
            best_model = model

    # Print the best parameters
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Predict again using the best parameters (re-evaluate)
    y_test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Training set evaluation
    y_train_pred = best_model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print(f"\nBest Model Training Set MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Best Model Testing Set MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

    return best_model, y_test_pred, test_mse, test_r2, best_params

# Train and optimize the GBDT model
best_gbdt_model, gbdt_pred, gbdt_mse, gbdt_r2, best_params = optimize_gbdt(X_train, y_train, X_test, y_test,
                                                                            n_combinations=200)

print(f"\nFinal Testing Set R² for the Best Model: {gbdt_r2:.4f}")


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
    cv_model = GradientBoostingRegressor(
        **best_params,
        random_state=SEED
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