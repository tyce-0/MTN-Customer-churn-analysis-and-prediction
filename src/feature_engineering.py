"""
Feature Engineering Module
Handles encoding, scaling, and feature transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


class FeatureEngineering:
    """Feature engineering and transformation"""
    
    def __init__(self, df, target_column='Customer Churn Status'):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.scaler = None
        self.X = None
        self.y = None
        
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\nEncoding categorical features...")
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        for col in categorical_cols:
            if col in ['Customer ID', 'Full Name', 'Reasons for Churn']:
                continue
                
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded: {col}")
        
        # Encode target variable
        if self.target_column in self.df.columns:
            le_target = LabelEncoder()
            self.df[self.target_column] = le_target.fit_transform(self.df[self.target_column])
            self.label_encoders[self.target_column] = le_target
            print(f"Encoded target: {self.target_column}")
        
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features"""
        print("\nCreating interaction features...")
        
        if 'Total Revenue' in self.df.columns and 'Customer Tenure in months' in self.df.columns:
            self.df['Revenue_per_Month'] = self.df['Total Revenue'] / (self.df['Customer Tenure in months'] + 1)
            print("Created: Revenue_per_Month")
        
        if 'Data Usage' in self.df.columns and 'Customer Tenure in months' in self.df.columns:
            self.df['Data_Usage_per_Month'] = self.df['Data Usage'] / (self.df['Customer Tenure in months'] + 1)
            print("Created: Data_Usage_per_Month")
        
        if 'Total Revenue' in self.df.columns and 'Number of Times Purchased' in self.df.columns:
            self.df['Avg_Purchase_Value'] = self.df['Total Revenue'] / (self.df['Number of Times Purchased'] + 1)
            print("Created: Avg_Purchase_Value")
        
        return self.df
    
    def prepare_features_target(self, drop_columns=None):
        """Prepare features and target variable"""
        if drop_columns is None:
            drop_columns = []
        
        print("\nPreparing features and target...")
        
        # Drop datetime columns explicitly
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Identify columns to drop
        columns_to_drop = [
            self.target_column,
            'Customer ID',
            'Full Name',
            'Date of Purchase',
            'Reasons for Churn'
        ] + datetime_cols + drop_columns
        
        # Remove duplicates and columns that don't exist
        columns_to_drop = list(set([col for col in columns_to_drop if col in self.df.columns]))
        
        print(f"Dropping columns: {columns_to_drop}")
        
        # Prepare X and y
        self.X = self.df.drop(columns=columns_to_drop)
        self.y = self.df[self.target_column]
        
        # Ensure all columns are numeric
        non_numeric = self.X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric:
            print(f"\nWarning: Non-numeric columns found: {non_numeric}")
            print("Dropping non-numeric columns...")
            self.X = self.X.drop(columns=non_numeric)
        
        print(f"\nFeatures prepared: {self.X.shape}")
        print(f"Target prepared: {self.y.shape}")
        print(f"Feature columns: {self.X.columns.tolist()}")
        
        return self.X, self.y
    
    def scale_features(self, method='standard'):
        """Scale numerical features"""
        print(f"\nScaling features using {method} scaler...")
        
        if self.X is None:
            raise ValueError("Please run prepare_features_target() first")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        X_scaled = self.scaler.fit_transform(self.X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index)
        
        print(f"Features scaled: {X_scaled.shape}")
        
        return X_scaled
    
    def save_encoders(self, filepath):
        """Save label encoders and scaler"""
        encoders_dict = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.X.columns.tolist() if self.X is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoders_dict, f)
        
        print(f"\nEncoders saved to: {filepath}")
    
    def load_encoders(self, filepath):
        """Load label encoders and scaler"""
        with open(filepath, 'rb') as f:
            encoders_dict = pickle.load(f)
        
        self.label_encoders = encoders_dict['label_encoders']
        self.scaler = encoders_dict['scaler']
        
        print(f"\nEncoders loaded from: {filepath}")
        
        return encoders_dict