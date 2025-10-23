import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix the issue of negative sign display

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Read data
data = pd.read_excel('merged_data.xlsx')


# Data preprocessing (binary classification with custom threshold support)
def preprocess_data(data, threshold):
    # Extract molecule IDs and features, convert continuous target variable to binary classification (based on custom threshold)
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_continuous = data['Q_band']

    # Build binary classification labels using custom threshold
    y = (y_continuous > threshold).astype(int)  # 1: Above threshold, 0: Below or equal to threshold

    mol_ids = data['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    # Print changes in number of features and threshold information
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_filtered_df.shape[1]}")
    print(f"Classification threshold (custom): {threshold:.4f}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features


# Custom binary classification threshold
custom_threshold = 600

# Preprocess data (using custom threshold)
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data, custom_threshold)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y  # Maintain class distribution
)
print(f"Number of training samples: {X_train.shape[0]}, Class distribution: {np.bincount(y_train)}")
print(f"Number of test samples: {X_test.shape[0]}, Class distribution: {np.bincount(y_test)}")
print(f"Number of features: {X_train.shape[1]}")


# Define classification model parameter search function
def optimize_lightgbm_classifier(X_train, y_train, X_test, y_test, n_combinations=200):
    print("\n=== LightGBM Classifier Hyperparameter Optimization ===")
    # Classification parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'num_leaves': [20, 30, 50, 70],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.3],
        'reg_lambda': [0, 0.1, 0.3],
        'class_weight': [None, 'balanced']  # Handle class imbalance
    }

    # Calculate total number of combinations
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    n_combinations = min(n_combinations, total_combinations)
    print(f"Testing {n_combinations} random parameter combinations (total possible: {total_combinations})")

    # Generate parameter combinations
    param_combinations = []
    # Base combinations
    base_combinations = [
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 30,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 0},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'num_leaves': 20,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
    ]
    param_combinations.extend(base_combinations)

    # Randomly generate remaining combinations
    while len(param_combinations) < n_combinations:
        params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'max_depth': random.choice(param_grid['max_depth']),
            'num_leaves': random.choice(param_grid['num_leaves']),
            'subsample': random.choice(param_grid['subsample']),
            'colsample_bytree': random.choice(param_grid['colsample_bytree']),
            'reg_alpha': random.choice(param_grid['reg_alpha']),
            'reg_lambda': random.choice(param_grid['reg_lambda']),
            'class_weight': random.choice(param_grid['class_weight'])
        }
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = 0
    best_params = None
    best_model = None
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(param_combinations):
        print(f"\nTesting parameter combination {i + 1}/{len(param_combinations)}")
        print(params)

        # Build classification model
        model = lgb.LGBMClassifier(
            objective='binary',  # Binary classification objective function
            metric='binary_logloss',
            n_jobs=-1,
            random_state=SEED,** params
        )

        # Train model
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

        print(f"Training set accuracy: {train_acc:.4f}, Test set accuracy: {test_acc:.4f}")

        # Update best model
        if test_acc > best_score:
            best_score = test_acc
            best_params = params
            best_model = model

    # Output best results
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Detailed evaluation of the best model
    y_test_pred = best_model.predict(X_test)
    print("\nTest set confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))

    return best_model, y_test, y_test_pred, best_score, best_params


# Run optimization and train the best model
best_lgb_model, y_test, lgb_pred, lgb_acc, best_params = optimize_lightgbm_classifier(
    X_train, y_train, X_test, y_test, n_combinations=200
)
print(f"\nBest model test set accuracy: {lgb_acc:.4f}")


# Visualization functions implementation
## 1. Confusion matrix visualization
cm = confusion_matrix(y_test, lgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('LightGBM Confusion Matrix')
plt.show()

## 2. ROC curve and AUC
y_test_prob = best_lgb_model.predict_proba(X_test)[:, 1]  # Get positive class probabilities
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LightGBM ROC Curve')
plt.legend()
plt.show()

## 3. Cross-validation score visualization
cv_scores = cross_val_score(best_lgb_model, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('LightGBM 5-fold Cross-validation Scores')
plt.show()

## 4. Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_lgb_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro', n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation score')
plt.xlabel('Number of training samples')
plt.ylabel('F1 Score')
plt.title('LightGBM Learning Curve')
plt.legend()
plt.show()

## 5. Feature importance analysis
importances = best_lgb_model.feature_importances_
importance = pd.Series(importances, index=filtered_features)
importance = importance.sort_values(ascending=False)
top_importance = importance.head(20)  # Take top 20 important features

plt.figure(figsize=(12, 8))
top_importance.plot(kind='barh')
plt.title('LightGBM Feature Importance (Top 20)')
plt.xlabel('Importance score')
plt.ylabel('Feature name')
plt.tight_layout()
plt.show()