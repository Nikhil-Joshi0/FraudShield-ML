import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_time_features(df):
    print("Creating time-based features...")
    if 'timestamp' in df.columns:
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        # Drop original timestamp as Logistic Regression expects numerical values
        df = df.drop(columns=['timestamp'])
    return df

def create_interaction_features(df):
    print("Creating interaction features...")
    if 'oldbalanceOrg' in df.columns and 'amount' in df.columns:
        df['balance_diff'] = df['oldbalanceOrg'] - df['amount']
        # Feature identifying if the transaction is emptying the account exactly
        df['is_emptying_account'] = (df['balance_diff'] == 0).astype(int)
    return df

def scale_features(X_train, X_test):
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    # We scale only the numerical columns that aren't one-hot encoded or binary targets
    # For simplicity, we can scale all features since StandardScaler handles binary well enough,
    # or selectively scale non-binary features. Let's scale all float/int features.
    
    # Fit on train, transform on both
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler

def engineering_pipeline(df):
    df = create_time_features(df)
    df = create_interaction_features(df)
    return df
