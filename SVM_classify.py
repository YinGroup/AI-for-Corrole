import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, learning_curve)
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of negative sign display

# 1. Data reading and exploration
df = pd.read_excel('D:/2025.7.25/Classify/1/.venv/merged_data.xlsx')
print("Basic information of the data:")
print(df.describe())
print("Missing value status:")
print(df.isnull().sum())

# 2. Feature and target variable processing
X = df.drop(columns=['Mol_ID', 'Q_band'])#Delete Mol_ID and Q_band, and divide the remaining variables into feature variables
#y = pd.cut(df['Q_band'], bins=[-np.inf, df['Q_band'].median(), np.inf], labels=[0, 1])#Set the median of Q_band as the dividing point
y = pd.cut(df['Q_band'], bins=[-np.inf, 600, np.inf], labels=[0, 1])#Custom dividing point

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build a pipeline with preprocessing and tuning
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

# 5. Parameter grid search
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 1],
    'svm__degree': [2, 3]
}
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='f1_macro',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# 6. Model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTest set accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# 7. Visualization functions
## 7.1 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

## 7.2 ROC curve and AUC
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

## 7.3 Cross-validation score visualization
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1_macro')
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores)
plt.ylim(0, 1)
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.title('5-fold Cross-Validation Scores')
plt.show()

## 7.4 Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='f1_macro'
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation Score')
plt.xlabel('Number of training samples')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

# 8. Feature importance analysis
if best_model.named_steps['svm'].kernel == 'linear':
    coefficients = best_model.named_steps['svm'].coef_[0]
    importance = pd.Series(abs(coefficients), index=X.columns)
    importance = importance.sort_values(ascending=False)
    top_importance = importance.head(20)

    plt.figure(figsize=(12, 8))
    top_importance.plot(kind='barh')
    plt.title('Feature Importance (Absolute Values of Linear SVM Coefficients)')
    plt.xlabel('Absolute Value of Coefficients')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()
else:
    result = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42
    )
    importance = pd.Series(result.importances_mean, index=X.columns)
    importance = importance.sort_values(ascending=False)
    top_importance = importance.head(20)

    plt.figure(figsize=(12, 8))
    top_importance.plot(kind='barh')
    plt.title('Feature Importance (Permutation Importance)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()