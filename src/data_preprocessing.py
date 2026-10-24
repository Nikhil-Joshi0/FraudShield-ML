import pandas as pd

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    print("Handling missing values...")
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df

def encode_categorical(df):
    print("Encoding categorical variables...")
    # Drop high cardinality columns to simplify Logistic Regression
    cols_to_drop = ['transaction_id', 'customer_id', 'merchant_id']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # One-hot encode transaction_type
    if 'transaction_type' in df.columns:
        df = pd.get_dummies(df, columns=['transaction_type'], drop_first=True)
        
    return df

def preprocess_pipeline(df):
    df = handle_missing_values(df)
    df = encode_categorical(df)
    return df

if __name__ == '__main__':
    # Test block
    df = load_data('../data/transactions.csv')
    df_preprocessed = preprocess_pipeline(df)
    print(df_preprocessed.head())
