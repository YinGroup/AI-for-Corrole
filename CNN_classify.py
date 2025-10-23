import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. Set random seeds font display
# ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of negative sign display

# ======================
# 2. Read and preprocess data (adapted for binary classification)
# ======================
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'merged_data.xlsx')
print(f'Attempting to load data file path: {file_path}')

if not os.path.exists(file_path):
    raise FileNotFoundError(f'Data file not found, please check the path: {file_path}')

data = pd.read_excel(file_path)


def preprocess_data(data, cutoff):
    X = data.drop(['Mol_ID', 'Q_band'], axis=1)
    y_continuous = data['Q_band']

    y = pd.cut(
        y_continuous,
        bins=[-np.inf, cutoff, np.inf],
        labels=[0, 1]
    ).astype(int)

    mol_ids = data['Mol_ID']

    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    feature_mask = selector.get_support()
    filtered_features = X.columns[feature_mask]
    X_filtered_df = pd.DataFrame(X_filtered, columns=filtered_features)

    X_filtered_df = X_filtered_df.dropna(axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered_df.columns)

    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after filtering: {X_scaled_df.shape[1]}")
    print(f"Binary classification distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    return X_scaled_df, y, mol_ids, scaler, filtered_features


cutoff_value = 600
X_processed, y, mol_ids, scaler, filtered_features = preprocess_data(
    data, cutoff=cutoff_value
)
num_classes = 2

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"\nNumber of training samples: {X_train.shape[0]}, Number of test samples: {X_test.shape[0]}")
print(f"Feature dimension: {X_train.shape[1]}")


# ======================
# 3. Define dataset and CNN classification model
# ======================
class CNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_filters_1=16, num_filters_2=32,
                 dropout=0.2, fc1_size=64, fc2_size=32):
        super(CNN1DClassifier, self).__init__()

        self.reshape = lambda x: x.unsqueeze(1)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=num_filters_1, out_channels=num_filters_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        conv_output_dim = (input_dim // 4) * num_filters_2

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_dim, fc1_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, 2)
        )

    def forward(self, x):
        x = self.reshape(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ======================
# 4. Define training and evaluation functions (enhanced version, supporting probability output)
# ======================
def train_cnn_classifier(X_train, y_train, X_val, y_val,
                         epochs=100, batch_size=32, learning_rate=0.001,
                         num_filters_1=16, num_filters_2=32,
                         dropout=0.2, fc1_size=64, fc2_size=32,
                         return_proba=False):
    train_dataset = CNNDataset(X_train, y_train)
    val_dataset = CNNDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DClassifier(
        input_dim=X_train.shape[1],
        num_filters_1=num_filters_1,
        num_filters_2=num_filters_2,
        dropout=dropout,
        fc1_size=fc1_size,
        fc2_size=fc2_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Training loss: {avg_loss:.4f}")

    model.eval()
    val_preds = []
    val_probs = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities

    val_acc = accuracy_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds, average='binary', zero_division=1)
    val_recall = recall_score(y_val, val_preds, average='binary')
    val_f1 = f1_score(y_val, val_preds, average='binary')

    if return_proba:
        return model, val_acc, val_precision, val_recall, val_f1, val_probs
    return model, val_acc, val_precision, val_recall, val_f1


# ======================
# 5. Hyperparameter search
# ======================
def search_hyperparams(X_train, y_train, X_test, y_test):
    X_sub_train, X_val, y_sub_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    param_grid = {
        'epochs': [100, 150],
        'batch_size': [16, 32],
        'learning_rate': [0.001, 0.0005],
        'num_filters_1': [16, 32],
        'num_filters_2': [32, 64],
        'dropout': [0.2, 0.3],
        'fc1_size': [64, 128],
        'fc2_size': [32, 64]
    }

    keys = list(param_grid.keys())
    best_f1 = 0.0
    best_params = None
    best_model = None

    all_combinations = list(itertools.product(*param_grid.values()))
    print(f"\n=== Starting hyperparameter search, total {len(all_combinations)} combinations ===")

    for idx, combination in enumerate(all_combinations, 1):
        current_params = {k: v for k, v in zip(keys, combination)}
        print(f"\nCombination {idx}/{len(all_combinations)} - Parameters: {current_params}")

        model, val_acc, val_precision, val_recall, val_f1 = train_cnn_classifier(
            X_sub_train, y_sub_train, X_val, y_val, **current_params
        )

        print(f"Validation set performance - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_params = current_params

    print("\n=== Hyperparameter search completed ===")
    print(f"Best parameters: {best_params}")
    print(f"Best validation set F1 score: {best_f1:.4f}")

    # Retrain and get test set probabilities
    final_model, _, _, _, _, test_probs = train_cnn_classifier(
        X_train, y_train, X_test, y_test,** best_params, return_proba=True
    )

    # Calculate test set metrics
    test_preds = np.array([1 if p > 0.5 else 0 for p in test_probs])  # Convert probabilities to labels
    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, average='binary', zero_division=1)
    test_recall = recall_score(y_test, test_preds, average='binary')
    test_f1 = f1_score(y_test, test_preds, average='binary')

    return final_model, best_params, test_preds, test_probs, test_acc, test_precision, test_recall, test_f1


# ======================
# 6. Cross-validation function (for visualization)
# ======================
def cnn_cross_val_score(X, y, params, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        _, _, _, _, val_f1 = train_cnn_classifier(
            X_tr, y_tr, X_val, y_val, **params
        )
        scores.append(val_f1)

    return np.array(scores)


# ======================
# 7. Learning curve function (for visualization)
# ======================
def cnn_learning_curve(X, y, params, train_sizes=np.linspace(0.1, 1.0, 5), cv=5):
    train_scores = []
    valid_scores = []

    for train_size in train_sizes:
        fold_train = []
        fold_valid = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

        for train_idx, val_idx in skf.split(X, y):
            # Take training data proportionally
            n_train = int(len(train_idx) * train_size)
            X_tr, X_val = X.iloc[train_idx[:n_train]], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx[:n_train]], y.iloc[val_idx]

            # Train and evaluate
            _, tr_acc, _, _, tr_f1 = train_cnn_classifier(
                X_tr, y_tr, X_tr, y_tr,** params  # Evaluate training score on the training set itself
            )
            _, val_acc, _, _, val_f1 = train_cnn_classifier(
                X_tr, y_tr, X_val, y_val, **params
            )

            fold_train.append(tr_f1)
            fold_valid.append(val_f1)

        train_scores.append(np.mean(fold_train))
        valid_scores.append(np.mean(fold_valid))

    return train_sizes, np.array(train_scores), np.array(valid_scores)


# ======================
# 8. Execute training and visualization
# ======================
if __name__ == "__main__":
    # Train model and get results
    best_model, best_params, test_preds, test_probs, test_acc, test_precision, test_recall, test_f1 = search_hyperparams(
        X_train, y_train, X_test, y_test
    )

    # Output final test set results
    print("\n=== Best model performance on test set ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Best hyperparameters: {best_params}")

    # Save model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'scaler': scaler,
        'filtered_features': filtered_features,
        'cutoff': cutoff_value
    }, 'best_cnn_binary_classifier.pth')
    print("\nModel saved to 'best_cnn_binary_classifier.pth'")

    # ======================
    # Visualization section
    # ======================
    # 1. Confusion matrix visualization
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # 2. Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, zero_division=0))

    # 3. ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # 4. Cross-validation scores visualization
    cv_scores = cnn_cross_val_score(X_processed, y, best_params, cv=5)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=range(1, 6), y=cv_scores)
    plt.ylim(0, 1)
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('5-fold Cross-validation Scores')
    plt.show()

    # 5. Learning curve
    train_sizes, train_scores, valid_scores = cnn_learning_curve(
        X_train, y_train, best_params,
        train_sizes=np.linspace(0.1, 1.0, 5), cv=5
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, valid_scores, label='Validation Score')
    plt.xlabel('Proportion of Training Samples')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()