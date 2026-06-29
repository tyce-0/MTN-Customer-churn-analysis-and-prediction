"""
Main Pipeline Script
Orchestrates the entire ML workflow
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaning
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
from src.model_building import ModelBuilder
from src.model_evaluation import ModelEvaluation
from src.hyperparameter_tuning import HyperparameterTuning


def create_output_directories():
    """Create output directories for models and plots"""
    directories = ['models', 'plots', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def main():
    """Main pipeline execution"""
    
    print("\n" + "="*70)
    print(" " * 15 + "MTN CUSTOMER CHURN PREDICTION")
    print(" " * 20 + "ML Pipeline Execution")
    print("="*70)
    
    # Create output directories
    create_output_directories()
    
    # ========================================
    # STEP 1: DATA INGESTION
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: DATA INGESTION")
    print("="*70)
    
    data_path = os.path.join('data', 'mtn_customer_churn.csv')
    ingestion = DataIngestion(data_path)
    df = ingestion.load_data()
    
    # Display basic info
    print("\nDataset Information:")
    info = ingestion.get_basic_info()
    print(f"Shape: {info['shape']}")
    print(f"Total Missing Values: {sum(info['missing_values'].values())}")
    print(f"Duplicates: {info['duplicates']}")
    
    print("\nSample Data:")
    print(ingestion.display_sample(5))
    
    print("\nTarget Distribution:")
    print(ingestion.get_target_distribution())
    
    # ========================================
    # STEP 2: DATA CLEANING
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: DATA CLEANING AND TRANSFORMATION")
    print("="*70)
    
    cleaner = DataCleaning(df)
    
    # Handle missing values
    df_cleaned = cleaner.handle_missing_values(strategy='auto')
    
    # Handle duplicates
    df_cleaned = cleaner.handle_duplicates(keep='first')
    
    # Convert data types
    df_cleaned = cleaner.convert_data_types()
    
    # Create derived features
    df_cleaned = cleaner.create_derived_features()
    
    # Handle outliers in numerical columns
    numerical_cols = ['Age', 'Total Revenue', 'Data Usage', 'Customer Tenure in months']
    df_cleaned = cleaner.handle_outliers(numerical_cols, method='iqr')
    
    df_cleaned = cleaner.get_cleaned_data()
    
    # ========================================
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    eda = EDA(df_cleaned, target_column='Customer Churn Status')
    eda.generate_eda_report(save_dir='plots')
    
    # ========================================
    # STEP 4: FEATURE ENGINEERING
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*70)
    
    feature_eng = FeatureEngineering(df_cleaned, target_column='Customer Churn Status')
    
    # Encode categorical features
    df_encoded = feature_eng.encode_categorical_features()
    
    # Create interaction features
    df_encoded = feature_eng.create_interaction_features()
    
    # Prepare features and target sets
    X, y = feature_eng.prepare_features_target()  
    
    # Verify no datetime columns remain
    print(f"\nData types in X:\n{X.dtypes.value_counts()}")
    
    # Scale features
    X_scaled = feature_eng.scale_features(method='standard')
    
    # Save encoders
    feature_eng.save_encoders('models/encoders.pkl')
    
    print(f"\nFinal feature set: {X_scaled.shape[1]} features")
    print(f"Feature names: {X_scaled.columns.tolist()}")
    
    # ========================================
    # STEP 5: MODEL BUILDING
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: MODEL BUILDING")
    print("="*70)
    
    model_builder = ModelBuilder(X_scaled, y, test_size=0.2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = model_builder.split_data(stratify=True)
    
    # Apply SMOTE to handle class imbalance
    X_train_resampled, y_train_resampled = model_builder.apply_smote(
        sampling_strategy='auto',
        k_neighbors=5
    )
    
    # Initialize and train models
    model_builder.initialize_models()
    trained_models = model_builder.train_all_models()
    
    # Cross-validation
    cv_results = model_builder.cross_validate_models(cv=5)
    
    # Feature importance
    feature_importance = model_builder.get_feature_importance('Random Forest', top_n=15)
    
    # ========================================
    # STEP 6: MODEL EVALUATION
    # ========================================
    print("\n" + "="*70)
    print("STEP 6: MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluation()
    comparison_df = evaluator.evaluate_all_models(trained_models, X_test, y_test)
    
    # Save comparison results
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nModel comparison saved to 'results/model_comparison.csv'")
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(save_dir='plots')
    
    # Get best model
    best_model_name, best_metrics = evaluator.get_best_model(metric='F1-Score')
    
    # Save best model (before tuning)
    model_builder.save_model(best_model_name, f'models/{best_model_name.replace(" ", "_")}_base.pkl')
    
    # ========================================
    # STEP 7: HYPERPARAMETER TUNING
    # ========================================
    print("\n" + "="*70)
    print("STEP 7: HYPERPARAMETER TUNING")
    print("="*70)
    
    # Note: Grid search can be time-consuming. Use random search for faster results
    tuner = HyperparameterTuning(X_train, y_train, X_test, y_test)
    
    print("\nPerforming Random Search (faster)...")
    best_models = tuner.tune_all_models(method='random', n_iter=30)
    
    # Compare tuned models
    tuned_comparison = tuner.compare_tuned_models()
    tuned_comparison.to_csv('results/tuned_model_comparison.csv', index=False)
    
    # Get overall best model
    best_tuned_model_name, best_tuned_model = tuner.get_best_model_overall(metric='test_score')
    
    # Save best tuned model
    tuner.save_best_model(best_tuned_model_name, f'models/{best_tuned_model_name.replace(" ", "_")}_tuned.pkl')
    
    # ========================================
    # STEP 8: FINAL EVALUATION
    # ========================================
    print("\n" + "="*70)
    print("STEP 8: FINAL EVALUATION OF TUNED MODELS")
    print("="*70)
    
    final_evaluator = ModelEvaluation()
    final_comparison = final_evaluator.evaluate_all_models(best_models, X_test, y_test)
    
    # Save final results
    final_comparison.to_csv('results/final_model_comparison.csv', index=False)
    
    # Generate final evaluation report
    final_evaluator.generate_evaluation_report(save_dir='plots')
    
    # ========================================
    # PIPELINE COMPLETE
    # ========================================
    print("\n" + "="*70)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("- Models saved in 'models/' directory")
    print("- Plots saved in 'plots/' directory")
    print("- Results saved in 'results/' directory")
    print("\nNext Steps:")
    print("- Review the model evaluation results")
    print("- Use the best model for predictions")
    print("- Run the Streamlit app: streamlit run app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()