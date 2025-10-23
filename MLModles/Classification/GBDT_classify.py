import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix the issue of negative sign display

# Set random seeds to ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Read data
data = pd.read_excel('merged_data.xlsx')


# Data preprocessing
def preprocess_data(data, threshold=None):
    # Extract molecule IDs and target values
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_continuous = data['Q_band']

    # Binarize the target variable
    if threshold is None:
        threshold = y_continuous.median()
    y = (y_continuous >= threshold).astype(int)  # 1 indicates above threshold, 0 indicates below
    mol_ids = data['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    filtered_features = X_filtered_df.columns

    # Print the change in the number of features
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_filtered_df.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=filtered_features)  # Now the number of column names matches the data

    return X_scaled_df, y, mol_ids, scaler, filtered_features, threshold


# Call the preprocessing function
# Replace with a specific threshold (set according to the actual value range of Q_band)
custom_threshold = 600
X_processed, y, mol_ids, scaler, filtered_features, threshold = preprocess_data(data, threshold=custom_threshold)
print(f"Classification threshold: {threshold:.4f}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y  # Use stratify for classification tasks to maintain class distribution
)
print(f"Number of training samples: {X_train.shape[0]} (Positive class ratio: {y_train.mean():.2f})")
print(f"Number of test samples: {X_test.shape[0]} (Positive class ratio: {y_test.mean():.2f})")
print(f"Number of features: {X_train.shape[1]}")


# Define the classification model parameter optimization function
def optimize_gbdt_classifier(X_train, y_train, X_test, y_test, n_combinations=200):
    print("\n=== GBDT Classification Model Hyperparameter Optimization ===")

    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'loss': ['log_loss', 'exponential']
    }

    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"Total number of possible parameter combinations: {total_combinations}")
    n_combinations = min(n_combinations, total_combinations)
    print(f"Will test {n_combinations} random parameter combinations")

    # Basic parameter combinations
    param_combinations = [
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5,
         'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.8, 'loss': 'log_loss'},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3,
         'min_samples_split': 3, 'min_samples_leaf': 2, 'subsample': 0.7, 'loss': 'log_loss'}
    ]

    while len(param_combinations) < n_combinations:
        params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'max_depth': random.choice(param_grid['max_depth']),
            'min_samples_split': random.choice(param_grid['min_samples_split']),
            'min_samples_leaf': random.choice(param_grid['min_samples_leaf']),
            'subsample': random.choice(param_grid['subsample']),
            'loss': random.choice(param_grid['loss'])
        }
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}")
        print(params)

        model = GradientBoostingClassifier(
            **params,
            random_state=SEED
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)

        results.append({
            'params': params,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc
        })

        print(f"Training set accuracy: {train_acc:.4f}")
        print(f"Test set accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print(f"Test set F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

        if test_f1 > best_score:
            best_score = test_f1
            best_params = params
            best_model = model

    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    print("\nBest model test set evaluation:")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

    return best_model, y_test, y_test_pred, y_test_proba, best_params


# Train and optimize the classification model
best_gbdt_model, y_test, gbdt_pred, gbdt_proba, best_params = optimize_gbdt_classifier(
    X_train, y_train, X_test, y_test, n_combinations=200
)


# Visualization functions implementation
## 1. Confusion matrix visualization
cm = confusion_matrix(y_test, gbdt_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('GBDT Confusion Matrix')
plt.show()

## 2. ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, gbdt_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GBDT ROC Curve')
plt.legend()
plt.show()

## 3. Cross-validation score visualization
cv_scores = cross_val_score(best_gbdt_model, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('GBDT 5-Fold Cross-Validation Scores')
plt.show()

## 4. Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_gbdt_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro', n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation Score')
plt.xlabel('Number of Training Samples')
plt.ylabel('F1 Score')
plt.title('GBDT Learning Curve')
plt.legend()
plt.show()

## 5. Feature importance analysis
importances = best_gbdt_model.feature_importances_
importance = pd.Series(importances, index=filtered_features)
importance = importance.sort_values(ascending=False)
top_importance = importance.head(20)  # Take top 20 important features

plt.figure(figsize=(12, 8))
top_importance.plot(kind='barh')
plt.title('GBDT Feature Importance (Top 20)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()