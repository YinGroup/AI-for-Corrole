import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix the issue of negative sign display

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Read data
data = pd.read_excel(r'D:\2025.7.25\Classify\1\.venv\merged_data.xlsx')


# Data preprocessing (modified to binary classification, supporting custom thresholds)
def preprocess_data(data, threshold):
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_continuous = data['Q_band']

    # Binary classification division: ≤ threshold is 0, > threshold is 1
    y = (y_continuous > threshold).astype(int)
    print(f"Binary classification threshold: {threshold} (≤ is 0, > is 1)")

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

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features


# Custom division threshold
user_threshold = 600

# Preprocess data (using binary classification)
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data, user_threshold)

# Split into training set and test set (stratified sampling, suitable for binary classification)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"Number of training set samples: {X_train.shape[0]}, Number of test set samples: {X_test.shape[0]}")
print(f"Training set class distribution: {pd.Series(y_train).value_counts()}")


# Classification model parameter optimization
def optimize_decision_tree_classifier(X_train, y_train, X_test, y_test, n_combinations=200):
    print("\n=== Decision Tree Classifier Hyperparameter Optimization ===")
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    total_combinations = np.prod([len(values) for values in param_grid.values()])
    n_combinations = min(n_combinations, total_combinations)
    print(f"Testing {n_combinations} parameter combinations (total possible: {total_combinations})")

    # Generate random parameter combinations
    param_combinations = []
    while len(param_combinations) < n_combinations:
        params = {
            'max_depth': random.choice(param_grid['max_depth']),
            'min_samples_split': random.choice(param_grid['min_samples_split']),
            'min_samples_leaf': random.choice(param_grid['min_samples_leaf']),
            'max_features': random.choice(param_grid['max_features']),
            'criterion': random.choice(param_grid['criterion'])
        }
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")
        model = DecisionTreeClassifier(
            **params,
            random_state=SEED
        )
        model.fit(X_train, y_train)

        # Evaluation
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

    # Best model evaluation
    print("\n=== Best Parameters ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    y_test_pred = best_model.predict(X_test)
    print("\nTest set confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))

    return best_model, y_test_pred, best_score, best_params


# Train the optimized binary classification model
best_clf, y_pred, best_acc, best_params = optimize_decision_tree_classifier(
    X_train, y_train, X_test, y_test, n_combinations=200
)
print(f"\nBest model test set accuracy: {best_acc:.4f}")


# Visualization functions
## 1. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Decision Tree Model Confusion Matrix')
plt.show()

## 2. ROC curve and AUC
y_prob = best_clf.predict_proba(X_test)[:, 1]  # Get the probability of positive class
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree Model ROC Curve')
plt.legend()
plt.show()

## 3. Cross-validation score visualization
cv_scores = cross_val_score(best_clf, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('Decision Tree Model 5-Fold Cross-Validation Scores')
plt.show()

## 4. Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_clf, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro'
)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation Score')
plt.xlabel('Number of Training Samples')
plt.ylabel('F1 Score')
plt.title('Decision Tree Model Learning Curve')
plt.legend()
plt.show()

## 5. Feature importance analysis
importance = pd.Series(best_clf.feature_importances_, index=filtered_features)
importance = importance.sort_values(ascending=False)
top_importance = importance.head(20)  # Take top 20 important features

plt.figure(figsize=(12, 8))
top_importance.plot(kind='barh')
plt.title('Decision Tree Model Feature Importance (Top 20)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()