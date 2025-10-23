import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier  # Replace original import
import random


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix the display issue of negative signs

# Set random seeds to ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load data
data = pd.read_excel('merged_data.xlsx')


# Data preprocessing
def preprocess_data(data, threshold=None):
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_reg = data['Q_band']

    if threshold is None:
        threshold = y_reg.median()
        threshold_type = "median"
    else:
        threshold_type = "custom value"

    y = (y_reg > threshold).astype(int)
    print(f"Classification threshold ({threshold_type}): {threshold:.4f}")
    print(f"Class distribution: {pd.Series(y).value_counts(normalize=True)}")

    mol_ids = data['Mol_ID']

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    # Remove columns with missing values
    X_filtered_df = X_filtered_df.dropna(axis=1)

    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_filtered_df.shape[1]}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    return X_scaled_df, y, mol_ids, scaler, filtered_features


# Function to create model
def create_model(hidden_units, input_dim):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Define function for parameter search of classification model
def optimize_nnr_classifier(X_train, y_train, X_test, y_test, hidden_units_list):
    print("\n=== Neural Network Classifier Hyperparameter Optimization ===")
    best_score = -np.inf
    best_hidden_units = None
    best_model = None
    best_preds = None
    best_probs = None
    results = []

    for i, hidden_units in enumerate(hidden_units_list):
        print(f"\nTesting parameter combination {i + 1}/{len(hidden_units_list)}")
        print(f"Number of hidden layer units: {hidden_units}")

        model = create_model(hidden_units, X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Prediction
        y_train_pred_proba = model.predict(X_train).flatten()
        y_train_pred = (y_train_pred_proba > 0.5).astype(int)
        y_test_pred_proba = model.predict(X_test).flatten()
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)

        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        results.append({
            'hidden_units': hidden_units,
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        })

        print(
            f"Training set - Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(
            f"Test set - Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        if test_f1 > best_score:
            best_score = test_f1
            best_hidden_units = hidden_units
            best_model = model
            best_preds = y_test_pred
            best_probs = y_test_pred_proba

    # Evaluate best model
    y_test_pred = best_preds
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print("\n=== Best Model ===")
    print(f"Number of hidden layer units: {best_hidden_units}")
    print(
        f"Test set Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 score: {test_f1:.4f}")

    return best_model, best_hidden_units, y_test, best_preds, best_probs, test_acc, test_precision, test_recall, test_f1


# Preprocess data
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(data, threshold=600)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"Number of training samples: {X_train.shape[0]}, Number of test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Define parameter search range
hidden_units_list = [16, 32, 64, 128, 256]

# Optimize and train classification model
best_model, best_hidden, y_test, y_pred, y_prob, acc, precision, recall, f1 = optimize_nnr_classifier(
    X_train, y_train, X_test, y_test, hidden_units_list
)

print(f"\nFinal best model performance - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}")

# Visualization functions
## 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

## 2. ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

## 3. Cross-Validation Scores
keras_model = KerasClassifier(
    model=create_model,
    hidden_units=best_hidden,
    input_dim=X_processed.shape[1],
    epochs=50,
    batch_size=32,
    verbose=0
)
cv_scores = cross_val_score(keras_model, X_processed, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('5-Fold Cross-Validation Scores')
plt.show()

## 4. Learning Curve
train_sizes, train_scores, valid_scores = learning_curve(
    keras_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
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