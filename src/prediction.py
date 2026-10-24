import sys
import os
import pandas as pd
import joblib

# Add the project root to the python path so imports from 'src' work when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import engineering_pipeline

def predict_fraud(data_dict, model_path='outputs/logistic_model.pkl', scaler_path='outputs/scaler.pkl'):
    """
    Takes a dictionary of single transaction data, loads the model, and predicts if it is fraud.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or Scaler not found at {model_path} / {scaler_path}. Please run main.py first.")
        return None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Apply identical preprocessing & engineering steps (need to handle missing columns if any)
    # 1. Preprocess
    df = preprocess_pipeline(df)
    
    # 2. Engineering
    df = engineering_pipeline(df)
    
    # Ensure all columns required by the model are present (e.g., from one-hot encoding)
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Rearrange columns to match scaler
    df = df[expected_cols]
    
    # 3. Scale
    df_scaled = scaler.transform(df)
    
    # 4. Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    return {
        'prediction': "Fraud" if prediction == 1 else "Legitimate",
        'probability': probability
    }

if __name__ == '__main__':
    # Sample test inference
    test_transaction = {
        'transaction_id': 'TXN_TEST_1',
        'timestamp': '2026-01-01 12:00:00',
        'customer_id': 'C1234',
        'merchant_id': 'M123',
        'transaction_type': 'TRANSFER',
        'amount': 25000.0,
        'oldbalanceOrg': 25000.0,
    }
    print("Testing single prediction...")
    result = predict_fraud(test_transaction)
    print(f"Prediction Result: {result}")
