"""
Hyperparameter Tuning Module
Implements Grid Search and Random Search for model optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from typing import Dict, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuning:
    """Class to perform hyperparameter tuning"""
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize HyperparameterTuning
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_models = {}
        self.tuning_results = {}
    
    def get_param_grids(self) -> Dict:
        """
        Get parameter grids for different models
        
        Returns:
            Dict: Dictionary of parameter grids
        """
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [1000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'max_features': ['sqrt', 'log2']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
        }
        
        return param_grids
    
    def get_random_param_distributions(self) -> Dict:
        """
        Get parameter distributions for random search
        
        Returns:
            Dict: Dictionary of parameter distributions
        """
        from scipy.stats import randint, uniform
        
        param_distributions = {
            'Random Forest': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'Gradient Boosting': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.7, 0.3)
            },
            'XGBoost': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 12),
                'min_child_weight': randint(1, 10),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3),
                'gamma': uniform(0, 0.5)
            }
        }
        
        return param_distributions
    
    def tune_model_grid_search(self, model_name: str, model, param_grid: Dict,
                               cv: int = 5, scoring: str = 'f1',
                               n_jobs: int = -1) -> Tuple[object, Dict]:
        """
        Tune model using Grid Search
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            param_grid (Dict): Parameter grid
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_jobs (int): Number of parallel jobs
        
        Returns:
            Tuple[object, Dict]: Best model and results
        """
        print(f"\n" + "="*50)
        print(f"GRID SEARCH - {model_name}")
        print("="*50)
        print(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        # Test score
        test_score = grid_search.score(self.X_test, self.y_test)
        print(f"Test Score: {test_score:.4f}")
        
        self.best_models[model_name] = grid_search.best_estimator_
        self.tuning_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_score': test_score,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        return grid_search.best_estimator_, self.tuning_results[model_name]
    
    def tune_model_random_search(self, model_name: str, model, param_distributions: Dict,
                                 n_iter: int = 50, cv: int = 5, scoring: str = 'f1',
                                 n_jobs: int = -1) -> Tuple[object, Dict]:
        """
        Tune model using Random Search
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            param_distributions (Dict): Parameter distributions
            n_iter (int): Number of iterations
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_jobs (int): Number of parallel jobs
        
        Returns:
            Tuple[object, Dict]: Best model and results
        """
        print(f"\n" + "="*50)
        print(f"RANDOM SEARCH - {model_name}")
        print("="*50)
        print(f"Testing {n_iter} random combinations...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest Parameters: {random_search.best_params_}")
        print(f"Best CV Score: {random_search.best_score_:.4f}")
        
        # Test score
        test_score = random_search.score(self.X_test, self.y_test)
        print(f"Test Score: {test_score:.4f}")
        
        self.best_models[model_name] = random_search.best_estimator_
        self.tuning_results[model_name] = {
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'test_score': test_score,
            'cv_results': pd.DataFrame(random_search.cv_results_)
        }
        
        return random_search.best_estimator_, self.tuning_results[model_name]
    
    def tune_all_models(self, method: str = 'grid', n_iter: int = 50) -> Dict:
        """
        Tune all models
        
        Args:
            method (str): Tuning method ('grid' or 'random')
            n_iter (int): Number of iterations for random search
        
        Returns:
            Dict: Dictionary of best models
        """
        print("\n" + "="*70)
        print(" " * 20 + f"HYPERPARAMETER TUNING ({method.upper()})")
        print("="*70)
        
        # Get models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        if method == 'grid':
            param_grids = self.get_param_grids()
            
            for model_name, model in models.items():
                if model_name in param_grids:
                    self.tune_model_grid_search(
                        model_name, model, param_grids[model_name]
                    )
        
        elif method == 'random':
            param_distributions = self.get_random_param_distributions()
            
            for model_name, model in models.items():
                if model_name in param_distributions:
                    self.tune_model_random_search(
                        model_name, model, param_distributions[model_name], n_iter=n_iter
                    )
        
        print("\n" + "="*70)
        print(" " * 20 + "TUNING COMPLETED")
        print("="*70)
        
        return self.best_models
    
    def compare_tuned_models(self) -> pd.DataFrame:
        """
        Compare tuned models
        
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in self.tuning_results.items():
            comparison_data.append({
                'Model': model_name,
                'Best CV Score': results['best_cv_score'],
                'Test Score': results['test_score'],
                'Improvement': results['test_score'] - results['best_cv_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Score', ascending=False)
        
        print("\n" + "="*50)
        print("TUNED MODELS COMPARISON")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_best_model(self, model_name: str, path: str):
        """
        Save best tuned model
        
        Args:
            model_name (str): Name of the model
            path (str): Path to save the model
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in best models")
        
        # Save just the model object directly (not wrapped in dict)
        joblib.dump(self.best_models[model_name], path)
        
        # Optionally save metadata separately
        metadata_path = path.replace('.pkl', '_metadata.pkl')
        metadata = {
            'model_name': model_name,
            'best_params': self.tuning_results[model_name]['best_params'],
            'best_cv_score': self.tuning_results[model_name]['best_cv_score'],
            'test_score': self.tuning_results[model_name]['test_score']
        }
        joblib.dump(metadata, metadata_path)
        
        print(f"\n✅ {model_name} saved to {path}")
        print(f"   Metadata saved to {metadata_path}")
    
    def get_best_model_overall(self, metric: str = 'test_score') -> Tuple[str, object]:
        """
        Get the overall best model
        
        Args:
            metric (str): Metric to use ('test_score' or 'best_cv_score')
        
        Returns:
            Tuple[str, object]: Best model name and model object
        """
        best_model_name = None
        best_score = -1
        
        for model_name, results in self.tuning_results.items():
            score = results[metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        print(f"\n" + "="*50)
        print(f"OVERALL BEST MODEL (based on {metric})")
        print("="*50)
        print(f"Model: {best_model_name}")
        print(f"{metric}: {best_score:.4f}")
        print(f"Best Parameters: {self.tuning_results[best_model_name]['best_params']}")
        
        return best_model_name, self.best_models[best_model_name]