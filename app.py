"""
Streamlit Application for MTN Customer Churn Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MTN Churn Predictor",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        # Try to load tuned models first (skip metadata files)
        model_files = [
            'models/Gradient_Boosting_tuned.pkl',
            'models/Random_Forest_tuned.pkl',
            'models/XGBoost_tuned.pkl',
            'models/Gradient_Boosting_base.pkl',
            'models/Random_Forest_base.pkl',
            'models/XGBoost_base.pkl'
        ]
        
        model = None
        model_path = None
        
        for path in model_files:
            if os.path.exists(path) and '_metadata' not in path:
                loaded_obj = None
                # First try pickle
                try:
                    with open(path, 'rb') as f:
                        loaded_obj = pickle.load(f)
                    print(f"✅ Loaded with pickle: {path}")
                except Exception as e_pickle:
                    # If pickle fails, try joblib (common for sklearn/joblib saved files)
                    try:
                        loaded_obj = joblib.load(path)
                        print(f"✅ Loaded with joblib: {path}")
                    except Exception as e_joblib:
                        print(f"⚠️ Failed to load {path} with pickle ({e_pickle}) and joblib ({e_joblib})")
                        continue

                # Check if it's a valid model (has predict method)
                if hasattr(loaded_obj, 'predict'):
                    model = loaded_obj
                    print(f"✅ Using model: {path}")
                elif isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                    model = loaded_obj['model']
                    print(f"✅ Extracted model from dict structure: {path}")
                else:
                    print(f"⚠️ Skipping {path}: Not a valid model object")
                    continue

                model_path = path
                break
        
        if model is None:
            raise FileNotFoundError("No trained model found. Please run main.py first.")
        
        # Load encoders
        encoders_path = 'models/encoders.pkl'
        if not os.path.exists(encoders_path):
            raise FileNotFoundError("Encoders not found. Please run main.py first.")
            
        with open(encoders_path, 'rb') as f:
            encoders_dict = pickle.load(f)
        
        return model, encoders_dict, model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def prepare_input_data(input_dict, encoders_dict):
    """Prepare input data for prediction"""
    # Create DataFrame
    df = pd.DataFrame([input_dict])
    
    # Create ALL derived features (same as in feature_engineering.py)
    
    # 1. Revenue-based features
    if 'Total Revenue' in df.columns and 'Number of Times Purchased' in df.columns:
        df['Avg_Revenue_Per_Purchase'] = df['Total Revenue'] / (df['Number of Times Purchased'] + 1)
    
    if 'Total Revenue' in df.columns and 'Customer Tenure in months' in df.columns:
        df['Revenue_per_Month'] = df['Total Revenue'] / (df['Customer Tenure in months'] + 1)
        df['Monthly_Revenue'] = df['Total Revenue'] / (df['Customer Tenure in months'] + 1)
    
    # 2. Data usage features
    if 'Data Usage' in df.columns and 'Customer Tenure in months' in df.columns:
        df['Data_Usage_per_Month'] = df['Data Usage'] / (df['Customer Tenure in months'] + 1)
    
    if 'Total Revenue' in df.columns and 'Number of Times Purchased' in df.columns:
        df['Avg_Purchase_Value'] = df['Total Revenue'] / (df['Number of Times Purchased'] + 1)
    
    # 3. Age groups
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Age'], 
            bins=[0, 25, 35, 45, 55, 100], 
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        df['Age_Group'] = df['Age_Group'].astype(str)
    
    # 4. Tenure groups
    if 'Customer Tenure in months' in df.columns:
        df['Tenure_Group'] = pd.cut(
            df['Customer Tenure in months'],
            bins=[0, 6, 12, 24, 48, 120],
            labels=['0-6', '7-12', '13-24', '25-48', '48+']
        )
        df['Tenure_Group'] = df['Tenure_Group'].astype(str)
    
    # 5. Revenue groups
    if 'Total Revenue' in df.columns:
        df['Revenue_Group'] = pd.cut(
            df['Total Revenue'],
            bins=[0, 10000, 50000, 100000, 500000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        df['Revenue_Group'] = df['Revenue_Group'].astype(str)
    
    # Encode categorical features
    label_encoders = encoders_dict['label_encoders']
    
    for col, encoder in label_encoders.items():
        if col in df.columns and col != 'Customer Churn Status':
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError as e:
                # Handle unseen labels
                print(f"Warning: Unseen label in {col}, using default value")
                df[col] = 0
    
    # Get feature columns from encoders
    feature_columns = encoders_dict['feature_columns']
    
    # Ensure all required features exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    
    # Keep only features used in training, in the correct order
    df_features = df[feature_columns]
    
    # Scale features
    scaler = encoders_dict['scaler']
    df_scaled = scaler.transform(df_features)
    
    return pd.DataFrame(df_scaled, columns=feature_columns)

def main():
    st.title("📱 MTN Customer Churn Prediction")
    st.write("Predict whether a customer is likely to churn")
    
    # Load model and encoders
    model, encoders_dict, model_path = load_model_and_encoders()
    
    if model is None:
        st.error("Failed to load model. Please ensure the model is trained and saved.")
        st.info("💡 Run `python main.py` to train the models first.")
        return
    
    st.success(f"✅ Model loaded successfully from: `{model_path}`")
    
    # Create input form
    st.header("Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=16, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        state = st.selectbox("State", [
            "Lagos", "Abuja (FCT)", "Kano", "Rivers", "Oyo", "Kaduna",
            "Enugu", "Delta", "Edo", "Anambra", "Imo", "Ogun",
            "Plateau", "Cross River", "Bauchi", "Benue", "Kogi",
            "Ondo", "Ekiti", "Kwara", "Osun", "Adamawa", "Akwa Ibom",
            "Bayelsa", "Borno", "Ebonyi", "Gombe", "Jigawa",
            "Kebbi", "Nasarawa", "Niger", "Sokoto", "Taraba", "Yobe", "Zamfara"
        ])
    
    with col2:
        device = st.selectbox("MTN Device", [
            "Mobile SIM Card", "4G Router", "Broadband MiFi",
            "5G Broadband Router"
        ])
        subscription = st.selectbox("Subscription Plan", [
            "500MB Daily Plan", "1GB+1.5mins Daily Plan",
            "1.5GB 2-Day Plan", "2.5GB 2-Day Plan", "3.2GB 2-Day Plan",
            "7GB Monthly Plan", "10GB+10mins Monthly Plan",
            "12.5GB Monthly Plan", "20GB Monthly Plan", "25GB Monthly Plan",
            "65GB Monthly Plan", "165GB Monthly Plan",
            "30GB Monthly Broadband Plan", "60GB Monthly Broadband Plan",
            "120GB Monthly Broadband Plan", "200GB Monthly Broadband Plan",
            "150GB FUP Monthly Unlimited", "300GB FUP Monthly Unlimited",
            "450GB 3-Month Broadband Plan", "1.5TB Yearly Broadband Plan"
        ])
        satisfaction = st.slider("Satisfaction Rate", 1, 5, 3)
    
    with col3:
        tenure = st.number_input("Customer Tenure (months)", min_value=1, max_value=120, value=12)
        purchases = st.number_input("Number of Times Purchased", min_value=1, max_value=50, value=5)
        unit_price = st.number_input("Unit Price", min_value=100, max_value=200000, value=5000)
        data_usage = st.number_input("Data Usage (GB)", min_value=0.0, max_value=200.0, value=50.0)
    
    # Calculate derived features
    total_revenue = unit_price * purchases
    
    # Map satisfaction to review
    review_mapping = {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}
    customer_review = review_mapping[satisfaction]
    
    # Prepare input dictionary
    input_dict = {
        'Age': age,
        'State': state,
        'MTN Device': device,
        'Gender': gender,
        'Satisfaction Rate': satisfaction,
        'Customer Review': customer_review,
        'Customer Tenure in months': tenure,
        'Subscription Plan': subscription,
        'Unit Price': unit_price,
        'Number of Times Purchased': purchases,
        'Total Revenue': total_revenue,
        'Data Usage': data_usage
    }
    
    # Display summary
    st.header("Customer Summary")
    st.write(f"**Total Revenue:** ₦{total_revenue:,.2f}")
    st.write(f"**Average Revenue per Month:** ₦{total_revenue/(tenure+1):,.2f}")
    st.write(f"**Data Usage per Month:** {data_usage/(tenure+1):.2f} GB")
    
    # Prediction button
    if st.button("🔮 Predict Churn", type="primary"):
        with st.spinner("Making prediction..."):
            try:
                # Prepare input data
                X_input = prepare_input_data(input_dict, encoders_dict)
                
                # Make prediction
                prediction = model.predict(X_input)[0]
                prediction_proba = model.predict_proba(X_input)[0]
                
                # Display results
                st.header("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ **HIGH RISK OF CHURN**")
                        st.write(f"Churn Probability: **{prediction_proba[1]:.2%}**")
                    else:
                        st.success("✅ **LOW RISK OF CHURN**")
                        st.write(f"Retention Probability: **{prediction_proba[0]:.2%}**")
                
                with col2:
                    st.metric("Churn Risk Score", f"{prediction_proba[1]:.2%}")
                
                # Recommendations
                if prediction == 1:
                    st.header("📋 Recommendations")
                    st.write("""
                    - Offer personalized retention incentives
                    - Improve customer service experience
                    - Provide better data plans or discounts
                    - Conduct customer satisfaction survey
                    - Reach out proactively to address concerns
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please ensure all fields are filled correctly.")

if __name__ == "__main__":
    main()