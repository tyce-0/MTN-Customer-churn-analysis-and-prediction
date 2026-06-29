"""
Data Cleaning and Transformation Module
Handles missing values, duplicates, outliers, and data type conversions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class DataCleaning:
    """Class to handle data cleaning operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataCleaning with dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.df = df.copy()
        self.original_shape = df.shape
    
    def handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): Strategy for handling missing values
                           'auto' - automatic based on data type
                           'drop' - drop rows with missing values
                           'fill' - fill with appropriate values
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print(f"\nMissing values before cleaning:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        if strategy == 'auto':
            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            # For categorical columns, fill with mode or 'Unknown'
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    # Fill 'Reasons for Churn' with 'No Churn' for non-churned customers
                    if col == 'Reasons for Churn':
                        self.df[col].fillna('No Churn', inplace=True)
                    else:
                        mode_value = self.df[col].mode()
                        if len(mode_value) > 0:
                            self.df[col].fillna(mode_value[0], inplace=True)
                        else:
                            self.df[col].fillna('Unknown', inplace=True)
        
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
        
        print(f"\nMissing values after cleaning: {self.df.isnull().sum().sum()}")
        return self.df
    
    def handle_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Handle duplicate rows
        
        Args:
            subset (List[str]): Columns to consider for duplicates
            keep (str): Which duplicate to keep ('first', 'last', False)
        
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        duplicates_before = self.df.duplicated(subset=subset).sum()
        print(f"\nDuplicates before: {duplicates_before}")
        
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        
        duplicates_after = self.df.duplicated(subset=subset).sum()
        print(f"Duplicates after: {duplicates_after}")
        
        return self.df
    
    def handle_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            columns (List[str]): Columns to check for outliers
            method (str): Method to detect outliers ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        outliers_count = 0
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outliers_count += outliers
                
                # Cap outliers instead of removing
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"\nOutliers handled: {outliers_count}")
        return self.df
    
    def convert_data_types(self) -> pd.DataFrame:
        """
        Convert data types to appropriate formats
        
        Returns:
            pd.DataFrame: Dataframe with corrected data types
        """
        # Convert 'Customer Churn Status' to binary
        if 'Customer Churn Status' in self.df.columns:
            self.df['Customer Churn Status'] = self.df['Customer Churn Status'].map({'Yes': 1, 'No': 0})
        
        # Convert date column if exists
        if 'Date of Purchase' in self.df.columns:
            self.df['Date of Purchase'] = pd.to_datetime(self.df['Date of Purchase'], format='%b-%y', errors='coerce')
        
        return self.df
    
    def create_derived_features(self) -> pd.DataFrame:
        """
        Create derived features from existing columns
        
        Returns:
            pd.DataFrame: Dataframe with new features
        """
        # Average revenue per purchase
        if 'Total Revenue' in self.df.columns and 'Number of Times Purchased' in self.df.columns:
            self.df['Avg_Revenue_Per_Purchase'] = self.df['Total Revenue'] / (self.df['Number of Times Purchased'] + 1)
        
        # Customer lifetime value proxy
        if 'Total Revenue' in self.df.columns and 'Customer Tenure in months' in self.df.columns:
            self.df['Monthly_Revenue'] = self.df['Total Revenue'] / (self.df['Customer Tenure in months'] + 1)
        
        # Age groups
        if 'Age' in self.df.columns:
            self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                          bins=[0, 25, 35, 50, 100], 
                                          labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        return self.df
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned dataframe
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print(f"\nData cleaning summary:")
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        
        return self.df