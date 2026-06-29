"""
Exploratory Data Analysis Module
Performs statistical analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class EDA:
    """Class to perform Exploratory Data Analysis"""
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'Customer Churn Status'):
        """
        Initialize EDA with dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
        """
        self.df = df.copy()
        self.target_column = target_column
        
        # Set style for visualizations
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def statistical_summary(self) -> pd.DataFrame:
        """
        Generate statistical summary of the dataset
        
        Returns:
            pd.DataFrame: Statistical summary
        """
        print("\n" + "="*50)
        print("STATISTICAL SUMMARY")
        print("="*50)
        
        summary = self.df.describe()
        print(summary)
        
        return summary
    
    def correlation_analysis(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Perform correlation analysis on numerical features
        
        Args:
            save_path (str): Path to save the plot
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show features most correlated with target
        if self.target_column in correlation_matrix.columns:
            print("\n" + "="*50)
            print("FEATURES MOST CORRELATED WITH CHURN")
            print("="*50)
            target_corr = correlation_matrix[self.target_column].sort_values(ascending=False)
            print(target_corr)
        
        return correlation_matrix
    
    def target_distribution(self, save_path: Optional[str] = None):
        """
        Visualize target variable distribution
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Count plot
        ax = sns.countplot(data=self.df, x=self.target_column, palette='Set2')
        plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Churn Status', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add percentage labels
        total = len(self.df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(f'{int(p.get_height())}\n({percentage})', (x, y), 
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, features: List[str], save_path: Optional[str] = None):
        """
        Analyze important features by their relationship with target
        
        Args:
            features (List[str]): List of features to analyze
            save_path (str): Path to save the plot
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature not in self.df.columns:
                continue
            
            ax = axes[idx]
            
            if not pd.api.types.is_numeric_dtype(self.df[feature]):
                # For categorical features
                churn_data = self.df.groupby([feature, self.target_column]).size().unstack(fill_value=0)
                churn_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                ax.set_title(f'{feature} vs Churn', fontsize=12, fontweight='bold')
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.legend(['No Churn', 'Churn'], loc='best')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # For numerical features, coerce to numeric and skip if invalid
                feature_values = pd.to_numeric(self.df[feature], errors='coerce')
                if feature_values.isna().all():
                    warnings.warn(f"Skipping feature '{feature}' for boxplot: no numeric values available.")
                    ax.set_visible(False)
                    continue
                plot_df = self.df[[feature, self.target_column]].copy()
                plot_df[feature] = feature_values
                plot_df.boxplot(column=feature, by=self.target_column, ax=ax)
                ax.set_title(f'{feature} vs Churn', fontsize=12, fontweight='bold')
                ax.set_xlabel('Churn Status', fontsize=10)
                ax.set_ylabel(feature, fontsize=10)
                plt.suptitle('')
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def distribution_plots(self, numerical_features: List[str], save_path: Optional[str] = None):
        """
        Create distribution plots for numerical features
        
        Args:
            numerical_features (List[str]): List of numerical features
            save_path (str): Path to save the plot
        """
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(numerical_features):
            if feature not in self.df.columns:
                continue
            
            ax = axes[idx]
            self.df[feature].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.axvline(self.df[feature].mean(), color='red', linestyle='--', 
                      label=f'Mean: {self.df[feature].mean():.2f}')
            ax.legend()
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_eda_report(self, save_dir: Optional[str] = None):
        """
        Generate comprehensive EDA report
        
        Args:
            save_dir (str): Directory to save plots
        """
        print("\n" + "="*70)
        print(" " * 20 + "EDA REPORT")
        print("="*70)
        
        # Statistical summary
        self.statistical_summary()
        
        # Target distribution
        print("\n" + "="*50)
        print("TARGET VARIABLE DISTRIBUTION")
        print("="*50)
        print(self.df[self.target_column].value_counts())
        self.target_distribution(save_path=f"{save_dir}/target_distribution.png" if save_dir else None)
        
        # Correlation analysis
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_features) > 1:
            self.correlation_analysis(save_path=f"{save_dir}/correlation_heatmap.png" if save_dir else None)
        
        # Important features analysis
        important_features = ['Satisfaction Rate', 'Customer Tenure in months', 
                             'Total Revenue', 'Data Usage', 'Age', 'Gender']
        available_features = [f for f in important_features if f in self.df.columns]
        
        if available_features:
            self.feature_importance_analysis(available_features, 
                                            save_path=f"{save_dir}/feature_importance.png" if save_dir else None)
        
        # Distribution plots
        if len(numerical_features) > 0:
            self.distribution_plots(numerical_features[:6], 
                                   save_path=f"{save_dir}/distributions.png" if save_dir else None)