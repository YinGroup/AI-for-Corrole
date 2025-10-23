import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix the negative sign display issue

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Read data
data = pd.read_excel(r'D:\2025.7.25\Classify\1\.venv\merged_data.xlsx')


# Data preprocessing (modified for binary classification task, supporting custom division threshold)
def preprocess_data(data, threshold):
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    # Binary classification division: According to the custom threshold, values greater than or equal to the threshold are 1, and less than are 0
    y = (data['Q_band'] >= threshold).astype(int)  # 1 indicates meeting the threshold condition, 0 indicates not
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


# Custom division threshold
custom_threshold = 600  # The value here can be adjusted according to actual needs

# Preprocess data (binary classification)
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data, threshold=custom_threshold)

# Stratified split into training and test sets (maintain class distribution)
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
train_idx, test_idx = next(strat_split.split(X_processed, y))
X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Number of training set samples: {X_train.shape[0]}")
print(f"Number of test set samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Binary classification distribution: {pd.Series(y).value_counts(normalize=True)}")  # Show the proportion of 0 and 1


# Hyperparameter optimization function (adapted for binary classification)
def optimize_et_classifier(X_train, y_train, X_test, y_test, n_combinations=200):
    print("\n=== Extra Trees Binary Classifier Hyperparameter Optimization ===")
    # Classifier parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']  # Binary classification may also have class imbalance
    }

    total_combinations = np.prod([len(values) for values in param_grid.values()])
    n_combinations = min(n_combinations, total_combinations)
    print(f"Testing {n_combinations} random parameter combinations")

    # Generate parameter combinations
    param_combinations = [
        {
            'n_estimators': 200,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'class_weight': None
        },
        {
            'n_estimators': 300,
            'max_depth': 7,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        }
    ]

    # Supplement with random combinations
    while len(param_combinations) < n_combinations:
        params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'max_depth': random.choice(param_grid['max_depth']),
            'min_samples_split': random.choice(param_grid['min_samples_split']),
            'min_samples_leaf': random.choice(param_grid['min_samples_leaf']),
            'max_features': random.choice(param_grid['max_features']),
            'class_weight': random.choice(param_grid['class_weight'])
        }
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = 0
    best_params = None
    best_model = None
    results = []

    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}")
        print(params)

        # Train classifier
        model = ExtraTreesClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1  # Only keep valid parameters
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

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

    # Evaluate best model
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]  # ExtraTrees can directly obtain probabilities
    print(f"\nBest model test set accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Add y_test (true labels) and y_test_prob (probabilities) to return values for visualization
    return best_model, y_test, y_test_pred, y_test_prob, best_score, best_params


# Optimize and train binary classifier
best_et_clf, y_test, et_pred, et_prob, et_acc, best_params = optimize_et_classifier(
    X_train, y_train, X_test, y_test, n_combinations=200
)
print(f"\nFinal test set accuracy: {et_acc:.4f}")


# Visualization implementation
## 1. Confusion matrix visualization
cm = confusion_matrix(y_test, et_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

## 2. ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, et_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

## 3. Cross-validation score visualization
cv_scores = cross_val_score(best_et_clf, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('5-fold Cross-Validation Scores')
plt.show()

## 4. Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_et_clf, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro', n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation Score')
plt.xlabel('Number of Training Samples')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

## 5. Feature importance analysis
importances = best_et_clf.feature_importances_
importance = pd.Series(importances, index=filtered_features)
importance = importance.sort_values(ascending=False)
top_importance = importance.head(20)  # Take top 20 important features

plt.figure(figsize=(12, 8))
top_importance.plot(kind='barh')
plt.title('Feature Importance (Extra Trees)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()