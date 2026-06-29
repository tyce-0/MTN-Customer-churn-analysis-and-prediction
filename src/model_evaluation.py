"""
Model Evaluation Module
Comprehensive model evaluation with multiple metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluation:
    """Class to evaluate classification models"""
    
    def __init__(self):
        """Initialize ModelEvaluation"""
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict:
        """
        Evaluate a single model with comprehensive metrics
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
        
        Returns:
            Dict: Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='binary', zero_division=0),
        }
        
        # AUC Score (if probability predictions available)
        if y_pred_proba is not None:
            metrics['AUC-Score'] = roc_auc_score(y_test, y_pred_proba)
        else:
            metrics['AUC-Score'] = None
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def evaluate_all_models(self, models: Dict, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all models and create comparison dataframe
        
        Args:
            models (Dict): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            pd.DataFrame: Comparison dataframe of all models
        """
        print("\n" + "="*70)
        print(" " * 25 + "MODEL EVALUATION")
        print("="*70)
        
        all_metrics = []
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate_model(model, X_test, y_test, name)
            all_metrics.append(metrics)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*70)
        print(" " * 20 + "MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None):
        """
        Plot confusion matrix for a model
        
        Args:
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        y_test = self.evaluation_results[model_name]['y_test']
        y_pred = self.evaluation_results[model_name]['y_pred']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed metrics
        print(f"\n" + "="*50)
        print(f"CONFUSION MATRIX - {model_name}")
        print("="*50)
        print(f"True Negatives: {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives: {cm[1][1]}")
    
    def plot_all_confusion_matrices(self, save_dir: Optional[str] = None):
        """
        Plot confusion matrices for all evaluated models
        
        Args:
            save_dir (str): Directory to save plots
        """
        n_models = len(self.evaluation_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            y_test = results['y_test']
            y_pred = results['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/all_confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, model_names: Optional[List[str]] = None, 
                      save_path: Optional[str] = None):
        """
        Plot ROC curves for models
        
        Args:
            model_names (List[str]): List of model names to plot. If None, plot all
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
            
            y_test = self.evaluation_results[model_name]['y_test']
            y_pred_proba = self.evaluation_results[model_name]['y_pred_proba']
            
            if y_pred_proba is None:
                continue
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, model_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None):
        """
        Plot Precision-Recall curves for models
        
        Args:
            model_names (List[str]): List of model names to plot. If None, plot all
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
            
            y_test = self.evaluation_results[model_name]['y_test']
            y_pred_proba = self.evaluation_results[model_name]['y_pred_proba']
            
            if y_pred_proba is None:
                continue
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(self, model_name: str):
        """
        Print detailed classification report
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        y_test = self.evaluation_results[model_name]['y_test']
        y_pred = self.evaluation_results[model_name]['y_pred']
        
        print(f"\n" + "="*50)
        print(f"CLASSIFICATION REPORT - {model_name}")
        print("="*50)
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Churn', 'Churn']))
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None):
        """
        Plot bar chart comparison of all metrics
        
        Args:
            save_path (str): Path to save the plot
        """
        metrics_data = []
        for model_name, results in self.evaluation_results.items():
            metrics_data.append(results['metrics'])
        
        df = pd.DataFrame(metrics_data)
        df = df.set_index('Model')
        
        # Remove None values
        df = df.fillna(0)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            df[metric].plot(kind='bar', ax=ax, color=colors[idx], alpha=0.8)
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        # Hide the extra subplot
        fig.delaxes(axes[5])
        
        plt.suptitle('Model Performance Metrics Comparison', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self, metric: str = 'F1-Score') -> Tuple[str, Dict]:
        """
        Get the best performing model based on a metric
        
        Args:
            metric (str): Metric to use for selection
        
        Returns:
            Tuple[str, Dict]: Best model name and its metrics
        """
        best_model_name = None
        best_score = -1
        best_metrics = None
        
        for model_name, results in self.evaluation_results.items():
            score = results['metrics'][metric]
            if score is not None and score > best_score:
                best_score = score
                best_model_name = model_name
                best_metrics = results['metrics']
        
        print(f"\n" + "="*50)
        print(f"BEST MODEL (based on {metric})")
        print("="*50)
        print(f"Model: {best_model_name}")
        print(f"{metric}: {best_score:.4f}")
        
        return best_model_name, best_metrics
    
    def generate_evaluation_report(self, save_dir: Optional[str] = None):
        """
        Generate comprehensive evaluation report with all visualizations
        
        Args:
            save_dir (str): Directory to save plots
        """
        print("\n" + "="*70)
        print(" " * 20 + "EVALUATION REPORT")
        print("="*70)
        
        # Plot all confusion matrices
        self.plot_all_confusion_matrices(save_dir=save_dir)
        
        # Plot ROC curves
        self.plot_roc_curve(save_path=f"{save_dir}/roc_curves.png" if save_dir else None)
        
        # Plot Precision-Recall curves
        self.plot_precision_recall_curve(save_path=f"{save_dir}/precision_recall_curves.png" if save_dir else None)
        
        # Plot metrics comparison
        self.plot_metrics_comparison(save_path=f"{save_dir}/metrics_comparison.png" if save_dir else None)
        
        # Print classification reports
        for model_name in self.evaluation_results.keys():
            self.print_classification_report(model_name)
        
        # Get best model
        self.get_best_model()