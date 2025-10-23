import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class CorrelationFeatureEliminator:
    """
    Feature Elimination class based on high correlation between features.
    It identifies and removes one feature from each pair that exceeds the specified threshold.
    """

    def __init__(self, threshold=0.7, cv=5, random_state=42):
        """
        Initialize parameters

        Parameters:
        -----------
        threshold : float, Correlation threshold (r) for feature elimination
        cv : int, Number of cross-validation folds
        random_state : int, Random seed
        """
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state

        # Initialize BayesianRidge for evaluation consistency
        self.evaluator_model = BayesianRidge(
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True
        )

        self.selected_features = None
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.df = None

    def load_data(self, file_path='merged_data.xlsx'):
        """
        Load data

        Parameters:
        -----------
        file_path : str, Path to the data file
        """
        print("Loading data...")
        self.df = pd.read_excel(file_path)
        print(f"Data loaded, shape: {self.df.shape}")

        # Separate features and target variable (using Q_band)
        self.X = self.df.drop(['Mol_ID', 'Q_band'], axis=1)
        self.y = self.df['Q_band']

        print(f"Number of features: {self.X.shape[1]}")
        print(f"Number of samples: {self.X.shape[0]}")
        print(f"Target variable range: {self.y.min():.3f} - {self.y.max():.3f}")

        return self.X, self.y

    def preprocess_data(self):
        """
        Data preprocessing: Impute missing values and standardize features.
        """
        print("Performing data preprocessing...")

        # Check for missing values
        missing_values = self.X.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found, processing...")
            print(f"Missing value statistics:\n{missing_values[missing_values > 0]}")
            # Impute missing values using the median
            self.X = self.X.fillna(self.X.median())

        # Standardize features
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.X.columns)

        print("Data preprocessing complete")

    def perform_correlation_selection(self):
        """
        Execute feature elimination based on correlation threshold.
        """
        print(f"Starting correlation elimination, threshold: {self.threshold}")

        # 1. Calculate the absolute correlation matrix
        corr_matrix = self.X_scaled.corr().abs()

        # 2. Select the upper triangle of the correlation matrix (excluding diagonal)
        # This prevents checking the same pair twice and correlation with self
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 3. Identify features to drop (one from each highly correlated pair)
        # It selects the feature that is the column name in the upper matrix.
        to_drop = [column for column in upper.columns if any(upper[column] >= self.threshold)]

        # 4. Determine the final selected features
        self.selected_features = [f for f in self.X.columns if f not in to_drop]

        print(f"Selection complete.")
        print(f"Features dropped due to high correlation: {len(to_drop)}")
        print(f"Number of selected features: {len(self.selected_features)}")

        return self.selected_features

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate model performance using the selected features.

        Parameters:
        -----------
        X_train, X_test : DataFrame, Scaled train/test set features (full set)
        y_train, y_test : Series, Train/test set target variable
        """
        print("\nEvaluating model performance with selected features...")

        # Filter train/test sets to include only selected features
        X_train_selected = X_train[self.selected_features]
        X_test_selected = X_test[self.selected_features]

        # Train the model
        self.evaluator_model.fit(X_train_selected, y_train)

        # Predict on the test set
        y_pred = self.evaluator_model.predict(X_test_selected)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Evaluation Results:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Coefficient of Determination (R²): {r2:.4f}")

        return mse, rmse, r2

    def cross_validation(self):
        """
        Cross-validation using the selected features and the evaluation model.
        """
        print("Performing cross-validation...")

        # Use the scaled data with only the selected features
        X_selected = self.X_scaled[self.selected_features]

        cv_scores = cross_val_score(
            self.evaluator_model,
            X_selected,
            self.y,
            cv=self.cv,
            scoring='r2'
        )

        print(f"Cross-Validation Results (R²):")
        print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Scores per fold: {cv_scores}")

        return cv_scores

    def save_processed_data(self, output_file='processed_data_corr.xlsx'):
        """
        Save the processed data (with selected features) to an Excel file.
        """
        print(f"Saving processed data to {output_file}...")

        # Create the processed DataFrame, maintaining original format
        processed_df = self.df.copy()

        # Keep only the selected features, plus Mol_ID and Q_band columns
        columns_to_keep = ['Mol_ID'] + self.selected_features + ['Q_band']
        processed_df = processed_df[columns_to_keep]

        # Save to Excel
        processed_df.to_excel(output_file, index=False)

        print(f"Processed data saved to {output_file}")
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Number of retained features: {len(self.selected_features)}")

        return processed_df

    def run_complete_analysis(self, test_size=0.2):
        """
        Run the complete analysis pipeline

        Parameters:
        -----------
        test_size : float, Test set ratio
        """
        print("=" * 50)
        print("Starting Correlation Feature Elimination Analysis")
        print("=" * 50)

        # 1. Load data
        self.load_data()

        # 2. Data preprocessing
        self.preprocess_data()

        # 3. Split into training and testing sets (using scaled data for consistency)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=self.random_state
        )

        # 4. Perform Correlation Selection
        self.perform_correlation_selection()

        # 5. Cross-validation
        cv_scores = self.cross_validation()

        # 6. Model evaluation
        mse, rmse, r2 = self.evaluate_model(X_train, X_test, y_train, y_test)

        # 7. Save processed data
        processed_df = self.save_processed_data()

        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print("=" * 50)

        return {
            'selected_features': self.selected_features,
            'cv_scores': cv_scores,
            'test_metrics': {'mse': mse, 'rmse': rmse, 'r2': r2},
            'processed_data': processed_df
        }


def main():
    """
    Main function to demonstrate the use of the CorrelationFeatureEliminator class
    """
    # Set parameters
    correlation_threshold = 0.7  # Adjust the correlation threshold as needed

    # Create the eliminator object
    corr_analyzer = CorrelationFeatureEliminator(
        threshold=correlation_threshold,
        cv=5,
        random_state=42
    )

    # Run complete analysis
    results = corr_analyzer.run_complete_analysis(test_size=0.2)

    # Print final result summary
    print(f"\nFinal Result Summary:")
    print(f"Original number of features: {len(corr_analyzer.X.columns)}")
    print(f"Number of selected features: {len(results['selected_features'])}")
    print(
        f"Feature reduction percentage: {(1 - len(results['selected_features']) / len(corr_analyzer.X.columns)) * 100:.1f}%")
    print(f"Cross-validation R²: {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})")
    print(f"Test Set R²: {results['test_metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()