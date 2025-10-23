import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
import random
from datetime import datetime


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix the negative sign display issue

# Set random seed to ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Construct absolute path
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'merged_data.xlsx')
print(f'Attempting to load data file path: {file_path}')

# Check file existence
if not os.path.exists(file_path):
    raise FileNotFoundError(f'Data file not found, please check the path: {file_path}')

data = pd.read_excel(file_path)


# Data preprocessing (for classification task)
def preprocess_data(data, cutoff):
    # Extract molecule ID and target value (discretize the continuous target variable into binary classification labels based on manually set cutoffs)
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_continuous = data['Q_band']

    y = pd.cut(y_continuous, bins=[-np.inf, cutoff, np.inf], labels=[0, 1])
    mol_ids = data['Mol_ID']

    # Check and remove columns with all zeros (features with zero variance)
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)

    # Get the names of retained features
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]

    # Convert the filtered data back to DataFrame
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    # Print the number of features before and after filtering
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_filtered_df.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features

cutoff_value = 600  # Manually set cutoff value, can be adjusted according to actual needs
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data, cutoff=cutoff_value)

# Split into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y  # Use stratify for classification task to maintain class distribution
)
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Class distribution: {np.bincount(y_train.astype(int))}")  # Binary classification distribution statistics


# Define the parameter search function for classification model
def optimize_adaboost_classifier(X_train, y_train, X_test, y_test, n_combinations=200):
    print("\n=== AdaBoost Classifier Hyperparameter Optimization ===")
    # Define parameter grid for classification task
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
        'algorithm': ['SAMME', 'SAMME.R'],  # Algorithm parameters specific to classification
        'base_estimator__max_depth': [1, 2, 3]  # Depth of the base decision tree
    }

    # Calculate the number of all possible combinations
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"Total number of possible parameter combinations: {total_combinations}")

    # Adjust the number of combinations to actually test
    n_combinations = min(n_combinations, total_combinations)
    print(f"Will test {n_combinations} random parameter combinations")

    # Generate random parameter combinations
    param_combinations = []
    # Base combinations
    base_combinations = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'algorithm': 'SAMME.R', 'base_estimator__max_depth': 1},
        {'n_estimators': 200, 'learning_rate': 0.05, 'algorithm': 'SAMME', 'base_estimator__max_depth': 2}
    ]
    param_combinations.extend(base_combinations)

    # Randomly generate remaining combinations
    while len(param_combinations) < n_combinations:
        params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'algorithm': random.choice(param_grid['algorithm']),
            'base_estimator__max_depth': random.choice(param_grid['base_estimator__max_depth'])
        }
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = 0
    best_params = None
    best_model = None
    best_pred = None
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}")
        print(params)

        # Create base classifier (decision tree)
        base_estimator = DecisionTreeClassifier(
            max_depth=params['base_estimator__max_depth'],
            random_state=SEED
        )

        # Create and train AdaBoost classifier
        model = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            algorithm=params['algorithm'],
            random_state=SEED
        )

        model.fit(X_train, y_train)

        # Evaluate training set
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Evaluate test set
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Store results
        results.append({
            'params': params,
            'train_acc': train_acc,
            'test_acc': test_acc
        })

        print(f"Training set accuracy: {train_acc:.4f}")
        print(f"Test set accuracy: {test_acc:.4f}")

        # Update best model
        if test_acc > best_score:
            best_score = test_acc
            best_params = params
            best_model = model
            best_pred = y_test_pred

    # Print best parameters
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Detailed evaluation of the best model
    print("\n=== Best Model Test Set Evaluation ===")
    print(f"Accuracy: {best_score:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred))

    return best_model, best_pred, best_score, best_params, X_train, y_train, X_test, y_test


# Train and optimize AdaBoost classifier
best_adaboost_model, adaboost_pred, adaboost_acc, best_params, X_train, y_train, X_test, y_test = optimize_adaboost_classifier(
    X_train, y_train, X_test, y_test
)
print(f"\nFinal test set accuracy of the best model: {adaboost_acc:.4f}")

# Visualization functions
## 1. Confusion Matrix
cm = confusion_matrix(y_test, adaboost_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('AdaBoost Confusion Matrix')
plt.show()

## 2. ROC Curve and AUC
y_prob = best_adaboost_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AdaBoost ROC Curve')
plt.legend()
plt.show()

## 3. Visualization of Cross-Validation Scores
cv_scores = cross_val_score(best_adaboost_model, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('AdaBoost 5-Fold Cross-Validation Scores')
plt.show()

## 4. Learning Curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_adaboost_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro'
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation Score')
plt.xlabel('Number of Training Samples')
plt.ylabel('F1 Score')
plt.title('AdaBoost Learning Curve')
plt.legend()
plt.show()

## 5. Feature Importance Analysis
importance = pd.Series(best_adaboost_model.feature_importances_, index=X_processed.columns)
importance = importance.sort_values(ascending=False)
top_importance = importance.head(20)  # Take top 20 important features

plt.figure(figsize=(12, 8))
top_importance.plot(kind='barh')
plt.title('AdaBoost Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()