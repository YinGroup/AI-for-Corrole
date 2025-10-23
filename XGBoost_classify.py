import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc)
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
import random
from datetime import datetime
import joblib  # For model saving


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of negative sign display

# Set random seed to ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Read data
data_path = os.path.join(current_dir, 'merged_data_kisc.xlsx')
try:
    data_df = pd.read_excel(data_path)
    print(f"Successfully read data, number of samples: {data_df.shape[0]}, number of features: {data_df.shape[1]}")
except Exception as e:
    print(f"Data reading failed: {e}")
    exit(1)


# Data preprocessing (classification task)
def preprocess_data(data_df, n_classes=2):
    # Extract molecule IDs and features (remove ID and target columns)
    X = data_df.drop(['Mol_ID', 'Q_band'], axis=1)

    # Convert continuous target variable Q_band to classification labels (binary classification)
    custom_threshold = 10
    y = (data_df['Q_band'] > custom_threshold).astype(int)  # 1 if greater than threshold, 0 otherwise

    mol_ids = data_df['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    filtered_features = X_filtered_df.columns

    # Print changes in the number of features
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_filtered_df.shape[1]}")

    # Standardize features (not necessary for tree models, can be adjusted as needed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features


# Data preprocessing
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data_df)

# Split into training and test sets (including molecule IDs for result analysis)
X_train, X_test, y_train, y_test, mol_ids_train, mol_ids_test = train_test_split(
    X_processed, y, mol_ids,
    test_size=0.3,
    random_state=SEED,
    stratify=y  # Maintain class distribution
)
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Class distribution of training set: {pd.Series(y_train).value_counts(normalize=True)}")


# Function for optimizing hyperparameters of XGBoost classifier
def optimize_xgboost_classifier(X_train, y_train, X_test, y_test, mol_ids_test, n_combinations=200):
    print("\n=== XGBoost Classifier Hyperparameter Optimization ===")

    # Classification parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]  # Handle class imbalance
    }

    # Calculate total combinations and limit the number of tests
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    n_combinations = min(n_combinations, total_combinations)
    print(f"Will test {n_combinations} random parameter combinations (total possible combinations: {total_combinations})")

    # Generate parameter combinations (including base combinations)
    param_combinations = [
        # Base classification parameter combinations
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'scale_pos_weight': 1},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 3,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1,
         'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)}
    ]

    # Supplement with random combinations (remove duplicates)
    while len(param_combinations) < n_combinations:
        params = {k: random.choice(v) for k, v in param_grid.items()}
        if params not in param_combinations:
            param_combinations.append(params)

    best_score = 0
    best_test_acc = 0
    best_params = None
    best_model = None
    best_test_pred = None
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(current_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)  # Create result directory

    for i, params in enumerate(param_combinations, 1):
        print(f"\nTesting parameter combination {i}/{len(param_combinations)}")
        print(params)

        # Build classification model
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # Objective function for binary classification
            eval_metric='logloss',
            **params,
            random_state=SEED
        )

        # Training and evaluation
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate classification metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        results.append({
            'params': params,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        })

        print(f"Training set accuracy: {train_acc:.4f} | Test set accuracy: {test_acc:.4f} | Test set F1: {test_f1:.4f}")

        # Update best model (using F1 score as the main metric)
        if test_f1 > best_score:
            best_score = test_f1
            best_test_acc = test_acc
            best_params = params
            best_model = model
            best_test_pred = y_test_pred

    # Output best model results
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    print(f"\nBest test set accuracy: {best_test_acc:.4f}")

    # Detailed evaluation of the best model
    print("\nTest set classification report:")
    print(classification_report(y_test, best_test_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, best_test_pred))

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(result_dir, f'xgb_param_results_{timestamp}.xlsx'), index=False)

    pred_df = pd.DataFrame({
        'Mol_ID': mol_ids_test,
        'true_label': y_test,
        'pred_label': best_test_pred
    })
    pred_df.to_excel(os.path.join(result_dir, f'xgb_predictions_{timestamp}.xlsx'), index=False)

    joblib.dump(best_model, os.path.join(result_dir, f'best_xgb_model_{timestamp}.pkl'))
    joblib.dump(scaler, os.path.join(result_dir, f'scaler_{timestamp}.pkl'))
    print(f"\nResults have been saved to the {result_dir} directory")

    return best_model, best_test_pred, best_params


# Train classification model
best_xgb_model, xgb_pred, best_params = optimize_xgboost_classifier(
    X_train, y_train, X_test, y_test, mol_ids_test, n_combinations=200
)

# Visualization implementation
## 1. Confusion matrix visualization
cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

## 2. ROC curve and AUC
y_test_prob = best_xgb_model.predict_proba(X_test)[:, 1]  # Get positive class probabilities
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

## 3. Cross-validation score visualization
cv_scores = cross_val_score(best_xgb_model, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('5-fold Cross-validation Scores')
plt.show()

## 4. Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_xgb_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro', n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation score')
plt.xlabel('Number of training samples')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

## 5. Feature importance analysis
importances = best_xgb_model.feature_importances_
importance = pd.Series(importances, index=filtered_features)
importance = importance.sort_values(ascending=False)
top_importance = importance.head(20)  # Take top 20 important features

plt.figure(figsize=(12, 8))
top_importance.plot(kind='barh')
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()