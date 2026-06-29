"""
Utility Functions
Helper functions for the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def print_section_header(title: str, width: int = 70):
    """
    Print a formatted section header
    
    Args:
        title (str): Section title
        width (int): Width of the header
    """
    print("\n" + "="*width)
    padding = (width - len(title)) // 2
    print(" " * padding + title)
    print("="*width)


def calculate_missing_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing value percentages
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Missing value statistics
    """
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values
    })
    
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    return missing_stats


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution"):
    """
    Plot class distribution
    
    Args:
        y (pd.Series): Target variable
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    counts = y.value_counts()
    
    plt.bar(counts.index, counts.values, color=['#2ecc71', '#e74c3c'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add percentages
    total = len(y)
    for i, (idx, count) in enumerate(counts.items()):
        percentage = f'{100 * count / total:.1f}%'
        plt.text(i, count, f'{count}\n({percentage})', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def create_age_groups(age: int) -> str:
    """
    Categorize age into groups
    
    Args:
        age (int): Age value
    
    Returns:
        str: Age group
    """
    if age < 25:
        return 'Young'
    elif age < 35:
        return 'Adult'
    elif age < 50:
        return 'Middle-aged'
    else:
        return 'Senior'


def calculate_churn_rate_by_feature(df: pd.DataFrame, feature: str, 
                                    target: str = 'Customer Churn Status') -> pd.DataFrame:
    """
    Calculate churn rate by feature
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature to analyze
        target (str): Target column
    
    Returns:
        pd.DataFrame: Churn rate statistics
    """
    churn_stats = df.groupby(feature)[target].agg([
        ('Total', 'count'),
        ('Churned', 'sum'),
        ('Churn_Rate', lambda x: x.sum() / len(x) * 100)
    ]).round(2)
    
    return churn_stats.sort_values('Churn_Rate', ascending=False)


def get_numerical_summary(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Get summary statistics for numerical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Specific columns to summarize
    
    Returns:
        pd.DataFrame: Summary statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = df[columns].describe().T
    summary['missing'] = df[columns].isnull().sum()
    summary['missing_pct'] = (df[columns].isnull().sum() / len(df) * 100).round(2)
    
    return summary


def save_predictions_with_details(df: pd.DataFrame, predictions: np.ndarray, 
                                  probabilities: np.ndarray, 
                                  filename: str = 'predictions.csv'):
    """
    Save predictions with original data
    
    Args:
        df (pd.DataFrame): Original dataframe
        predictions (np.ndarray): Predictions
        probabilities (np.ndarray): Prediction probabilities
        filename (str): Output filename
    """
    result_df = df.copy()
    result_df['Predicted_Churn'] = predictions
    result_df['Churn_Probability'] = probabilities
    result_df['Risk_Level'] = result_df['Churn_Probability'].apply(
        lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.4 else 'LOW'
    )
    
    result_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def format_currency(amount: float) -> str:
    """
    Format amount as Nigerian Naira
    
    Args:
        amount (float): Amount to format
    
    Returns:
        str: Formatted currency string
    """
    return f"₦{amount:,.2f}"


def calculate_business_metrics(df: pd.DataFrame, 
                               target_col: str = 'Customer Churn Status') -> Dict:
    """
    Calculate business metrics
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
    
    Returns:
        Dict: Business metrics
    """
    total_customers = len(df)
    churned_customers = df[target_col].sum()
    churn_rate = (churned_customers / total_customers) * 100
    
    if 'Total Revenue' in df.columns:
        total_revenue = df['Total Revenue'].sum()
        avg_revenue = df['Total Revenue'].mean()
        revenue_from_churned = df[df[target_col] == 1]['Total Revenue'].sum()
        potential_loss = revenue_from_churned
    else:
        total_revenue = avg_revenue = revenue_from_churned = potential_loss = 0
    
    metrics = {
        'Total Customers': total_customers,
        'Churned Customers': churned_customers,
        'Retained Customers': total_customers - churned_customers,
        'Churn Rate (%)': round(churn_rate, 2),
        'Retention Rate (%)': round(100 - churn_rate, 2),
        'Total Revenue': format_currency(total_revenue),
        'Average Revenue per Customer': format_currency(avg_revenue),
        'Revenue from Churned Customers': format_currency(revenue_from_churned),
        'Potential Revenue Loss': format_currency(potential_loss)
    }
    
    return metrics


def print_business_metrics(metrics: Dict):
    """
    Print business metrics in a formatted way
    
    Args:
        metrics (Dict): Business metrics dictionary
    """
    print_section_header("BUSINESS METRICS", width=60)
    
    for key, value in metrics.items():
        print(f"{key:.<40} {value}")
    
    print("="*60)