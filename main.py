import os
import joblib
from sklearn.model_selection import train_test_split

from src.data_generator import generate_mock_data
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import engineering_pipeline, scale_features
from src.model_training import apply_smote, train_logistic_regression
from src.evaluation import evaluate_model, plot_class_distribution

def main():
    print("Starting AI-Powered Financial Fraud Detection Pipeline...")
    
    data_path = 'data/transactions.csv'
   
    # 1. Generate Data if not exists
    if not os.path.exists(data_path):
        print("\n--- Phase 1: Data Generation ---")
        df = generate_mock_data(num_samples=15000, fraud_ratio=0.04, output_path=data_path)
    else:
        import pandas as pd
        print(f"\n--- Phase 1: Loading existing data from {data_path} ---")
        df = pd.read_csv(data_path)
        
    print(f"Dataset shape: {df.shape}")
    
    # 2. Preprocessing
    print("\n--- Phase 2: Preprocessing ---")
    df = preprocess_pipeline(df)
    print(f"Shape after preprocessing: {df.shape}")
    
    # 3. Feature Engineering
    print("\n--- Phase 3: Feature Engineering ---")
    df = engineering_pipeline(df)
    
    # Separate Target and Features
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Plot initial class distribution
    plot_class_distribution(y, output_dir='outputs', title="Initial Class Distribution")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 4. Handle Imbalance & Model Training
    print("\n--- Phase 4: Handle Imbalance and Train Model ---")
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    
    # Plot resampled distribution
    plot_class_distribution(y_train_resampled, output_dir='outputs', title="SMOTE Resampled Class Distribution")
    
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # Save model and scaler
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/logistic_model.pkl')
    joblib.dump(scaler, 'outputs/scaler.pkl')
    print("Model and Scaler saved to outputs/")
    
    # 5. Evaluation
    print("\n--- Phase 5: Evaluation ---")
    metrics = evaluate_model(model, X_test_scaled, y_test, output_dir='outputs')
    
    print("\nPipeline execution completed successfully.")
    print("Check the 'outputs/' directory for evaluation plots and saved models.")

if __name__ == '__main__':
    main()
