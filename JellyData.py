import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class DataCleaner:
    def __init__(self, file_path):
        """
        Initialize the DataCleaner with the input file
        """
        try:
            # Handle different file types
            if file_path.endswith('.csv'):
                self.original_df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.original_df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.original_df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")

            # Create a copy of the original dataframe
            self.cleaned_df = self.original_df.copy()

            # Configure logging
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)

        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            raise

    def analyze_null_values(self, plot=True):
        """
        Comprehensive analysis of null values
        """
        self.logger.info("Analyzing Null Values...")

        # Null value analysis
        null_analysis = pd.DataFrame({
            'Total Rows': len(self.cleaned_df),
            'Null Count': self.cleaned_df.isnull().sum(),
            'Null Percentage': (self.cleaned_df.isnull().sum() / len(self.cleaned_df) * 100).round(2),
            'Data Type': self.cleaned_df.dtypes
        })

        # Optional visualization
        if plot:
            plt.figure(figsize=(15, 6))
            sns.barplot(x=null_analysis.index, y=null_analysis['Null Percentage'])
            plt.title('Percentage of Null Values by Column')
            plt.xticks(rotation=90)
            plt.ylabel('Null Percentage')
            plt.tight_layout()
            plt.show()

        return null_analysis

    def remove_null_columns(self, threshold=0.5):
        """
        Remove columns with majority null values

        Args:
            threshold (float): Percentage of null values to drop column (default 50%)
        """
        self.logger.info(f"Removing columns with more than {threshold * 100}% null values...")

        # Calculate null percentages
        null_percentages = self.cleaned_df.isnull().mean()

        # Columns to drop
        columns_to_drop = null_percentages[null_percentages > threshold].index.tolist()

        # Log dropped columns
        for col in columns_to_drop:
            self.logger.info(f"Dropped: {col} - {null_percentages[col] * 100:.2f}% null")

        # Remove columns
        self.cleaned_df = self.cleaned_df.drop(columns=columns_to_drop)

        return self

    def handle_missing_values(self, method='mean'):
        """
        Handle missing values with different strategies

        Args:
            method (str): Imputation method ('mean', 'median', 'mode', 'drop')
        """
        self.logger.info(f"Handling missing values using {method} method...")

        # Identify numeric and categorical columns
        numeric_columns = self.cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_columns = self.cleaned_df.select_dtypes(include=['object']).columns

        # Handle missing values based on method
        if method == 'mean':
            self.cleaned_df[numeric_columns] = self.cleaned_df[numeric_columns].fillna(
                self.cleaned_df[numeric_columns].mean())
        elif method == 'median':
            self.cleaned_df[numeric_columns] = self.cleaned_df[numeric_columns].fillna(
                self.cleaned_df[numeric_columns].median())
        elif method == 'mode':
            # Numeric columns
            self.cleaned_df[numeric_columns] = self.cleaned_df[numeric_columns].fillna(
                self.cleaned_df[numeric_columns].mode().iloc[0])
            # Categorical columns
            self.cleaned_df[categorical_columns] = self.cleaned_df[categorical_columns].fillna(
                self.cleaned_df[categorical_columns].mode().iloc[0])
        elif method == 'drop':
            self.cleaned_df = self.cleaned_df.dropna()

        return self

    def remove_duplicates(self):
        """
        Remove duplicate rows
        """
        initial_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.drop_duplicates()

        duplicates_removed = initial_rows - len(self.cleaned_df)
        self.logger.info(f"Removed {duplicates_removed} duplicate rows")

        return self

    def clean_categorical_columns(self):
        """
        Clean and standardize categorical columns
        """
        self.logger.info("Cleaning categorical columns...")

        # Identify categorical columns
        categorical_columns = self.cleaned_df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            # Convert to lowercase
            self.cleaned_df[col] = self.cleaned_df[col].str.strip().str.lower()

            # Remove special characters
            self.cleaned_df[col] = self.cleaned_df[col].str.replace('[^a-zA-Z0-9\s]', '', regex=True)

        return self

    def detect_outliers(self, method='iqr'):
        """
        Detect outliers in numeric columns

        Args:
            method (str): Outlier detection method
        """
        self.logger.info("Detecting outliers...")

        numeric_columns = self.cleaned_df.select_dtypes(include=[np.number]).columns

        outliers_summary = {}

        for col in numeric_columns:
            Q1 = self.cleaned_df[col].quantile(0.25)
            Q3 = self.cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.cleaned_df[(self.cleaned_df[col] < lower_bound) | (self.cleaned_df[col] > upper_bound)]
            outliers_summary[col] = len(outliers)

            self.logger.info(f"{col}: {len(outliers)} outliers detected")

        return outliers_summary

    def save_cleaned_data(self, output_file='cleaned_dataset.csv'):
        """
        Save the cleaned dataset

        Args:
            output_file (str): Path to save cleaned dataset
        """
        try:
            self.cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
            self.logger.info(f"Cleaned data saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")

        return self

    def generate_report(self):
        """
        Generate a comprehensive data cleaning report
        """
        report = {
            'Original Rows': len(self.original_df),
            'Cleaned Rows': len(self.cleaned_df),
            'Original Columns': len(self.original_df.columns),
            'Cleaned Columns': len(self.cleaned_df.columns)
        }

        print("\n--- Data Cleaning Report ---")
        for key, value in report.items():
            print(f"{key}: {value}")

        return report


def main():
    try:
        # File path
        file_path = 'JellyData.csv'  # Update with the actual path to your CSV file

        # Create a DataCleaner instance
        cleaner = DataCleaner(file_path)

        # Analyze null values
        cleaner.analyze_null_values()

        # Remove columns with more than 50% null values
        cleaner.remove_null_columns(threshold=0.5)

        # Handle missing values
        cleaner.handle_missing_values(method='mean')

        # Remove duplicates
        cleaner.remove_duplicates()

        # Clean categorical columns
        cleaner.clean_categorical_columns()

        # Detect outliers
        outliers = cleaner.detect_outliers(method='iqr')

        # Save cleaned data
        cleaner.save_cleaned_data(output_file='cleaned_JEDI.csv')

        # Generate report
        cleaner.generate_report()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()