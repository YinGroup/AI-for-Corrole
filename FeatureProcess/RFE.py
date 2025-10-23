import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class BayesianRidgeRFE:
    """
    Recursive Feature Elimination class based on Bayesian Ridge Regression
    """

    def __init__(self, n_features_to_select=210, step=1, cv=5, random_state=42):
        """
        Initialize parameters

        Parameters:
        -----------
        n_features_to_select : int, Number of features to retain
        step : int, Number of features to remove at each iteration
        cv : int, Number of cross-validation folds
        random_state : int, Random seed
        """
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.cv = cv
        self.random_state = random_state

        # Initialize the model
        self.bayesian_ridge = BayesianRidge(
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True
        )

        # Initialize RFE
        self.rfe = RFE(
            estimator=self.bayesian_ridge,
            n_features_to_select=n_features_to_select,
            step=step
        )

        self.selected_features = None
        self.feature_ranking = None
        self.feature_scores = None
        self.scaler = StandardScaler()

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

        # Separate features and target variable
        self.X = self.df.drop(['Mol_ID', 'Q_band'], axis=1)  # Q_hand changed to Q_band
        self.y = self.df['Q_band']  # Q_hand changed to Q_band

        print(f"Number of features: {self.X.shape[1]}")
        print(f"Number of samples: {self.X.shape[0]}")
        print(f"Target variable range: {self.y.min():.3f} - {self.y.max():.3f}")

        return self.X, self.y

    def preprocess_data(self):
        """
        Data preprocessing
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

    def perform_rfe(self):
        """
        Execute Recursive Feature Elimination
        """
        print(f"Starting RFE, target number of features to retain: {self.n_features_to_select}")

        # Execute RFE
        self.rfe.fit(self.X_scaled, self.y)

        # Get selected features
        self.selected_features = self.X.columns[self.rfe.support_].tolist()
        self.feature_ranking = self.rfe.ranking_
        self.feature_scores = self.rfe.estimator_.coef_

        print(f"RFE complete, number of selected features: {len(self.selected_features)}")
        print(f"Selected features: {self.selected_features}")

        return self.selected_features

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance

        Parameters:
        -----------
        X_test : DataFrame, Test set features
        y_test : Series, Test set target variable
        """
        # Standardize the test set (using the full feature set)
        X_test_scaled = self.scaler.transform(X_test[self.X.columns])
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.X.columns)

        # Predict using RFE (RFE automatically selects features)
        y_pred = self.rfe.predict(X_test_scaled)

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
        Cross-validation
        """
        print("Performing cross-validation...")

        # Perform cross-validation using the selected features
        X_selected = self.X_scaled[self.selected_features]

        cv_scores = cross_val_score(
            self.rfe.estimator_,
            X_selected,
            self.y,
            cv=self.cv,
            scoring='r2'
        )

        print(f"Cross-Validation Results (R²):")
        print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Scores per fold: {cv_scores}")

        return cv_scores

    def get_feature_importance(self):
        """
        Get feature importance

        Returns:
        --------
        DataFrame: Feature importance ranking
        """
        # Get feature importance (absolute value of coefficients)
        feature_importance = np.abs(self.feature_scores)

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def get_feature_ranking(self):
        """
        Get feature ranking

        Returns:
        --------
        DataFrame: Feature ranking
        """
        # Create ranking DataFrame
        ranking_df = pd.DataFrame({
            'feature': self.X.columns,
            'ranking': self.rfe.ranking_
        }).sort_values('ranking')

        return ranking_df

    def save_processed_data(self, output_file='processed_data.xlsx'):
        """
        Save the processed data, maintaining the same format as the original data

        Parameters:
        -----------
        output_file : str, Output filename
        """
        print(f"Saving processed data to {output_file}...")

        # Create the processed DataFrame, maintaining original format
        processed_df = self.df.copy()

        # Keep only the selected feature columns, plus Mol_ID and Q_band columns
        columns_to_keep = ['Mol_ID'] + self.selected_features + ['Q_band']  # Q_hand changed to Q_band
        processed_df = processed_df[columns_to_keep]

        # Save to Excel
        processed_df.to_excel(output_file, index=False)

        print(f"Processed data saved to {output_file}")
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Number of retained features: {len(self.selected_features)}")

        return processed_df

    def save_results(self, output_file='rfe_results.xlsx'):
        """
        Save results to an Excel file

        Parameters:
        -----------
        output_file : str, Output filename
        """
        print(f"Saving results to {output_file}...")

        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': self.X.columns,
            'ranking': self.feature_ranking,
            'selected': self.rfe.support_
        })

        # Add coefficient information (only for selected features)
        coefficients = np.zeros(len(self.X.columns))
        coefficients[self.rfe.support_] = self.feature_scores
        results_df['coefficient'] = coefficients
        results_df['abs_coefficient'] = np.abs(coefficients)

        # Sort
        results_df = results_df.sort_values('ranking')

        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='RFE_Results', index=False)

            # Create a detailed table for selected features
            selected_df = results_df[results_df['selected'] == True].copy()
            selected_df = selected_df.sort_values('abs_coefficient', ascending=False)
            selected_df.to_excel(writer, sheet_name='Selected_Features', index=False)

        print(f"Results saved to {output_file}")

    def run_complete_analysis(self, test_size=0.2):
        """
        Run the complete analysis pipeline

        Parameters:
        -----------
        test_size : float, Test set ratio
        """
        print("=" * 50)
        print("Starting Bayesian Ridge Regression RFE Analysis")
        print("=" * 50)

        # 1. Load data
        self.load_data()

        # 2. Data preprocessing
        self.preprocess_data()

        # 3. Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=self.random_state
        )

        # 4. Perform RFE
        self.perform_rfe()

        # 5. Cross-validation
        cv_scores = self.cross_validation()

        # 6. Model evaluation
        # Note: X_test here is the scaled data used for splitting, but evaluate_model
        # expects the unscaled X_test (before RFE) for proper scaling/prediction flow.
        # However, the split was performed on X_scaled.
        # For demonstration purposes, we adjust the indexing of self.X and self.y
        # to ensure the correct features are passed to the evaluate_model.

        # Re-split using the unscaled, pre-imputation X data indices to get the correct X_test
        X_unscaled_train, X_unscaled_test, y_train_full, y_test_full = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )

        mse, rmse, r2 = self.evaluate_model(X_unscaled_test, y_test_full)

        # 7. Get feature information
        importance_df = self.get_feature_importance()
        ranking_df = self.get_feature_ranking()

        # 8. Save results
        self.save_results()

        # 9. Save processed data
        processed_df = self.save_processed_data()

        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print("=" * 50)

        return {
            'selected_features': self.selected_features,
            'importance_df': importance_df,
            'ranking_df': ranking_df,
            'cv_scores': cv_scores,
            'test_metrics': {'mse': mse, 'rmse': rmse, 'r2': r2},
            'processed_data': processed_df
        }


def main():
    """
    Main function to demonstrate the use of the BayesianRidgeRFE class
    """
    # Set parameters
    n_features_to_select = 74  # Number of features to retain, adjust as needed

    # Create RFE object
    rfe_analyzer = BayesianRidgeRFE(
        n_features_to_select=n_features_to_select,
        step=1,
        cv=5,
        random_state=42
    )

    # Run complete analysis
    results = rfe_analyzer.run_complete_analysis(test_size=0.2)

    # Print final result summary
    print(f"\nFinal Result Summary:")
    print(f"Original number of features: {len(rfe_analyzer.X.columns)}")
    print(f"Number of selected features: {len(results['selected_features'])}")
    print(
        f"Feature reduction percentage: {(1 - len(results['selected_features']) / len(rfe_analyzer.X.columns)) * 100:.1f}%")
    print(f"Cross-validation R²: {results['cv_scores'].mean():.4f} (+/- {results['cv_scores'].std() * 2:.4f})")
    print(f"Test Set R²: {results['test_metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()