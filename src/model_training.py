from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import pandas as pd

def apply_smote(X_train, y_train):
    print("Applying SMOTE to handle class imbalance...")
    print(f"Original class distribution:\n{y_train.value_counts()}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
    return X_resampled, y_resampled

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression model...")
    # Setting max_iter and class_weight just in case; since we use SMOTE, classes are balanced
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model
