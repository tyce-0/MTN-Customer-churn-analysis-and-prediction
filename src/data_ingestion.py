"""
Data Ingestion Module
Handles loading data from CSV files and basic data inspection
"""

import pandas as pd
import os
from typing import Tuple


class DataIngestion:
    """Class to handle data ingestion operations"""
    
    def __init__(self, file_path: str):
        """
        Initialize DataIngestion with file path
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded successfully!")
        print(f"Shape: {self.df.shape}")
        return self.df
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            dict: Dictionary containing basic dataset information
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
    
    def display_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Display sample rows from the dataset
        
        Args:
            n (int): Number of rows to display
            
        Returns:
            pd.DataFrame: Sample dataframe
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.head(n)
    
    def get_target_distribution(self, target_column: str = 'Customer Churn Status') -> pd.Series:
        """
        Get distribution of target variable
        
        Args:
            target_column (str): Name of target column
            
        Returns:
            pd.Series: Value counts of target variable
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df[target_column].value_counts()