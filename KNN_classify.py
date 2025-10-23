import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime

# Set random seeds to ensure reproducibility of results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Read data
data = pd.read_excel('merged_data_kisc.xlsx')


# Data preprocessing
def preprocess_data(data, threshold):
    # Extract molecule IDs and target values (divide the continuous variable Q_band into two classes based on the threshold)
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_continuous = data['Q_band']

    # Perform binary classification division based on custom threshold: >= threshold is 1, < threshold is 0
    y = (y_continuous >= threshold).astype(int)
    print(f"Binary classification division threshold: {threshold}")
    print(f"Binary classification sample distribution: Class 0-{np.sum(y == 0)}, Class 1-{np.sum(y == 1)}")

    mol_ids = data['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    # Print the change in the number of features
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_filtered_df.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features


# Custom binary classification division threshold
classification_threshold = 10

# Call the preprocessing function (for binary classification)
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(
    data,
    threshold=classification_threshold
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"Number of training set samples: {X_train.shape[0]}")
print(f"Number of testing set samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Training set class distribution: Class 0-{np.sum(y_train == 0)}, Class 1-{np.sum(y_train == 1)}")
print(f"Testing set class distribution: Class 0-{np.sum(y_test == 0)}, Class 1-{np.sum(y_test == 1)}")


# Define the classification model parameter optimization function (applicable to binary classification)
def optimize_knn_classifier(X_train, y_train, X_test, y_test, n_combinations=200):
    print("\n=== KNN Binary Classification Model Hyperparameter Optimization ===")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1=Manhattan distance, 2=Euclidean distance
    }

    total_combinations = np.prod([len(values) for values in param_grid.values()])
    n_combinations = min(n_combinations, total_combinations)
    print(f"Testing {n_combinations} random parameter combinations (total possible: {total_combinations})")

    # Basic parameter combinations + random combinations
    base_combinations = [
        {'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
        {'n_neighbors': 11, 'weights': 'distance', 'p': 2},
        {'n_neighbors': 7, 'weights': 'uniform', 'p': 1},
        {'n_neighbors': 9, 'weights': 'distance', 'p': 1}
    ]
    param_combinations = base_combinations
    while len(param_combinations) < n_combinations:
        params = {
            'n_neighbors': random.choice(param_grid['n_neighbors']),
            'weights': random.choice(param_grid['weights']),
            'p': random.choice(param_grid['p'])
        }
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}: {params}")
        # Build classification model
        model = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            p=params['p']
        )
        model.fit(X_train, y_train)

        # Prediction and evaluation
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Binary classification evaluation metrics (using binary averaging method)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        # Binary classification uses binary averaging, add zero_division=0 to avoid warnings
        test_precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)

        results.append({
            'params': params,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        })

        print(f"Training set accuracy: {train_acc:.4f}")
        print(f"Testing set accuracy: {test_acc:.4f}, F1 score: {test_f1:.4f}")

        # Update the best model (using testing set F1 score as the core indicator)
        if test_f1 > best_score:
            best_score = test_f1
            best_params = params
            best_model = model

    # Output the best results
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    y_test_pred = best_model.predict(X_test)
    print(f"\nBest model testing set performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred, average='binary'):.4f}")
    print(f"F1 score: {f1_score(y_test, y_test_pred, average='binary'):.4f}")

    return best_model, y_test_pred, best_params, results


# Train and optimize the KNN binary classifier
best_knn_clf, y_pred, best_params, results = optimize_knn_classifier(
    X_train, y_train, X_test, y_test, n_combinations=200
)