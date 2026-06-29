"""
Model Building Module
Handles model initialization, training, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import warnings
warnings.filterwarnings('ignore')


class ModelBuilder:
    """Model building and training"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.use_smote = False
        self.X_train_resampled = None
        self.y_train_resampled = None
        
    def split_data(self, stratify=True):
        """Split data into train and test sets"""
        print("\nSplitting data...")
        
        stratify_param = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"\nClass distribution in training set:")
        print(self.y_train.value_counts())
        print(f"\nClass distribution in test set:")
        print(self.y_test.value_counts())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smote(self, sampling_strategy='auto', k_neighbors=5):
        """Apply SMOTE to handle class imbalance"""
        print("\n" + "="*70)
        print("APPLYING SMOTE FOR CLASS IMBALANCE")
        print("="*70)
        
        if self.X_train is None:
            raise ValueError("Please run split_data() first")
        
        print(f"\nOriginal class distribution:")
        print(self.y_train.value_counts())
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )
        
        print(f"\nResampled class distribution:")
        print(pd.Series(self.y_train_resampled).value_counts())
        
        print(f"\nOriginal training set: {self.X_train.shape}")
        print(f"Resampled training set: {self.X_train_resampled.shape}")
        
        self.use_smote = True
        
        return self.X_train_resampled, self.y_train_resampled
    
    def initialize_models(self):
        """Initialize ML models"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, name, model):
        """Train a single model"""
        print(f"\nTraining {name}...")
        
        # Use resampled data if SMOTE was applied
        if self.use_smote and self.X_train_resampled is not None:
            X_train = self.X_train_resampled
            y_train = self.y_train_resampled
        else:
            X_train = self.X_train
            y_train = self.y_train
        
        model.fit(X_train, y_train)
        
        # Calculate training score
        train_score = model.score(X_train, y_train)
        print(f"{name} - Training Score: {train_score:.4f}")
        
        return model
    
    def train_all_models(self):
        """Train all initialized models"""
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        if not self.models:
            raise ValueError("Please run initialize_models() first")
        
        if self.X_train is None:
            raise ValueError("Please run split_data() first")
        
        for name, model in self.models.items():
            trained_model = self.train_model(name, model)
            self.trained_models[name] = trained_model
        
        print(f"\nAll {len(self.trained_models)} models trained successfully!")
        
        return self.trained_models
    
    def cross_validate_models(self, cv=5):
        """Perform cross-validation on all models"""
        print("\n" + "="*70)
        print("CROSS-VALIDATION")
        print("="*70)
        
        if not self.trained_models:
            raise ValueError("Please run train_all_models() first")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Use resampled data if SMOTE was applied
        if self.use_smote and self.X_train_resampled is not None:
            X_train = self.X_train_resampled
            y_train = self.y_train_resampled
        else:
            X_train = self.X_train
            y_train = self.y_train
        
        for name, model in self.trained_models.items():
            print(f"\nCross-validating {name}...")
            
            scores = cross_val_score(
                model, X_train, y_train,
                cv=skf,
                scoring='accuracy',
                n_jobs=-1
            )
            
            cv_results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            
            print(f"{name} - CV Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def get_feature_importance(self, model_name, top_n=10):
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X.columns
            
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop {top_n} Feature Importances for {model_name}:")
            print(feature_imp_df.head(top_n))
            
            return feature_imp_df
        else:
            print(f"{model_name} does not have feature_importances_ attribute")
            return None
    
    def save_model(self, model_name, filepath):
        """Save a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
        
        print(f"\nModel '{model_name}' saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"\nModel loaded from: {filepath}")
        return model