import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_mock_data(num_samples=15000, fraud_ratio=0.04, output_path='data/transactions.csv'):
    """
    Generates a realistic mock dataset for financial transactions with class imbalance.
    """
    print(f"Generating {num_samples} mock transactions...")
    np.random.seed(42)
    
    # Generate dates between Dec 2025 and Feb 2026
    start_date = datetime(2025, 12, 1)
    timestamps = [start_date + timedelta(minutes=int(np.random.randint(0, 131040))) for _ in range(num_samples)]
    timestamps.sort()
    
    # Basic fields
    customer_ids = [f'C{np.random.randint(1000, 9999)}' for _ in range(num_samples)]
    merchant_ids = [f'M{np.random.randint(100, 999)}' for _ in range(num_samples)]
    transaction_types = np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], size=num_samples)
    
    # Amounts and balances
    amounts = np.round(np.random.exponential(scale=200, size=num_samples), 2)
    oldbalanceOrg = np.round(np.random.uniform(0, 10000, size=num_samples), 2)
    
    # Fraud targets
    is_fraud = np.random.choice([0, 1], size=num_samples, p=[1 - fraud_ratio, fraud_ratio])
    
    # Apply synthetic patterns for fraud
    for i in range(num_samples):
        if is_fraud[i]:
            # Fraud consists mostly of TRANSFER and CASH_OUT
            transaction_types[i] = np.random.choice(['TRANSFER', 'CASH_OUT'])
            # Fraudulent amounts are typically higher or drain entire balance
            if np.random.rand() > 0.5:
                amounts[i] = oldbalanceOrg[i]
            else:
                amounts[i] = np.round(np.random.uniform(5000, 25000), 2)
            
    # Calculate new balance
    newbalanceOrig = oldbalanceOrg - amounts
    newbalanceOrig = np.where(newbalanceOrig < 0, 0, newbalanceOrig)
    
    df = pd.DataFrame({
        'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(num_samples)],
        'timestamp': timestamps,
        'customer_id': customer_ids,
        'merchant_id': merchant_ids,
        'transaction_type': transaction_types,
        'amount': amounts,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'is_fraud': is_fraud
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path}")
    print("\nClass Distribution:")
    print(df['is_fraud'].value_counts(normalize=True))
    
    return df

if __name__ == '__main__':
    generate_mock_data()
