import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataExplorer:
    """
    A comprehensive data exploration class for machine learning projects.
    
    This class provides methods for:
    - Basic data overview
    - Missing value analysis
    - Statistical analysis
    - Distribution analysis
    - Correlation analysis
    - Outlier detection
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        """
        Initialize DataExplorer with a dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str, optional): Name of target variable
        """
        self.df = df.copy()
        self.target_column = target_column
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
    def basic_info(self) -> None:
        """
        Display basic information about the dataset.
        """
        print("\n=== BASIC DATASET INFORMATION ===")
        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Types:")
        print(self.df.dtypes)
        print("\nMemory Usage:")
        print(self.df.memory_usage(deep=True))
        
    def missing_value_analysis(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Returns:
            pd.DataFrame: Missing value statistics
        """
        print("\n=== MISSING VALUE ANALYSIS ===")
        
        # Calculate missing value statistics
        missing_stats = pd.DataFrame({
            'Missing Values': self.df.isnull().sum(),
            'Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        }).sort_values('Percentage', ascending=False)
        
        # Only show columns with missing values
        missing_stats = missing_stats[missing_stats['Missing Values'] > 0]
        
        if len(missing_stats) > 0:
            print("\nColumns with missing values:")
            print(missing_stats)
            
            # Visualize missing values
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_stats.index, y='Percentage', data=missing_stats)
            plt.xticks(rotation=45)
            plt.title('Percentage of Missing Values by Column')
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo missing values found in the dataset.")
            
        return missing_stats
    
    def numerical_analysis(self) -> pd.DataFrame:
        """
        Perform statistical analysis on numerical columns.
        
        Returns:
            pd.DataFrame: Statistical summary of numerical columns
        """
        print("\n=== NUMERICAL COLUMNS ANALYSIS ===")
        
        # Basic statistics
        stats_df = self.df[self.numerical_columns].describe()
        print("\nNumerical Statistics:")
        print(stats_df)
        
        # Distribution plots
        for column in self.numerical_columns:
            plt.figure(figsize=(12, 4))
            
            # Create subplot for histogram
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45)
            
            # Create subplot for box plot
            plt.subplot(1, 2, 2)
            sns.boxplot(y=self.df[column])
            plt.title(f'Box Plot of {column}')
            
            plt.tight_layout()
            plt.show()
            
        return stats_df
    
    def categorical_analysis(self) -> Dict[str, pd.Series]:
        """
        Analyze categorical columns.
        
        Returns:
            Dict[str, pd.Series]: Value counts for each categorical column
        """
        print("\n=== CATEGORICAL COLUMNS ANALYSIS ===")
        
        value_counts_dict = {}
        
        for column in self.categorical_columns:
            print(f"\nValue counts for {column}:")
            value_counts = self.df[column].value_counts()
            print(value_counts)
            value_counts_dict[column] = value_counts
            
            # Visualize distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(y=column, data=self.df)
            plt.title(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()
            
        return value_counts_dict
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Analyze correlations between numerical variables.
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numerical_columns].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        if self.target_column:
            print(f"\nCorrelations with target variable ({self.target_column}):")
            target_corr = corr_matrix[self.target_column].sort_values(ascending=False)
            print(target_corr)
            
        return corr_matrix
    
    def detect_outliers(self, threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """
        Detect outliers in numerical columns using IQR method.
        
        Args:
            threshold (float): IQR multiplier for outlier detection
            
        Returns:
            Dict[str, np.ndarray]: Indices of outliers for each numerical column
        """
        print("\n=== OUTLIER DETECTION ===")
        
        outliers_dict = {}
        
        for column in self.numerical_columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = self.df[
                (self.df[column] < lower_bound) | 
                (self.df[column] > upper_bound)
            ].index
            
            if len(outliers) > 0:
                print(f"\nOutliers found in {column}:")
                print(f"Number of outliers: {len(outliers)}")
                print(f"Percentage of outliers: {(len(outliers)/len(self.df))*100:.2f}%")
                outliers_dict[column] = outliers
                
        return outliers_dict
    
    def generate_report(self) -> None:
        """
        Generate a complete exploratory data analysis report.
        """
        print("=== EXPLORATORY DATA ANALYSIS REPORT ===")
        
        # Run all analyses
        self.basic_info()
        self.missing_value_analysis()
        self.numerical_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.detect_outliers()

def main():
    """
    Example usage of DataExplorer class.
    """
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')
    
    # Initialize explorer with target column
    explorer = DataExplorer(df, target_column='median_house_value')
    
    # Generate complete report
    explorer.generate_report()

if __name__ == "__main__":
    main()
