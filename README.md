<<<<<<< HEAD
# AI-Powered Financial Fraud Detection System

**Duration**: Dec 2025 – Feb 2026

## Project Overview
This project involves an end-to-end Machine Learning pipeline designed to detect fraudulent financial transactions. It addresses common real-world financial data challenges, heavily focusing on handling imbalanced classes and building robust evaluation mechanisms. 

The primary model used is **Logistic Regression**, which provides explainability and acts as an excellent baseline for tabular structured data in financial domains.

## Methodology
The pipeline consists of modularized steps:
1. **Data Generation/Loading**: Uses synthesized mock data showing normal user patterns alongside anomalous fraudulent activities (e.g., account drains, high-amount transfers).
2. **Data Preprocessing**: Handling missing values, cleaning, and categorical encoding. High-cardinality IDs are dropped for linear model simplicity.
3. **Feature Engineering**: Creating time-based features and behavioral interaction features (e.g., `balance_diff`). Features are subsequently standardized using `StandardScaler`.
4. **Resampling**: Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to algorithmically balance the training data, solving the problem where the model becomes biased toward the non-fraud majority class.
5. **Model Training**: Logistic Regression trained on the geometrically balanced SMOTE data.
6. **Evaluation**: Instead of relying purely on accuracy, the model evaluates on Precision, Recall, and F1-score to reduce False Negatives (missed frauds). Confusion Matrix and ROC Curves visualize performance.

## Folder Structure
```
fraud detection system/
│
├── data/
│   └── transactions.csv         # Generated mock data
├── src/
│   ├── data_generator.py        # Generates synthetic structured datasets
│   ├── data_preprocessing.py    # Handling NaNs, encoding
│   ├── feature_engineering.py   # Synthesizing and scaling features
│   ├── model_training.py        # SMOTE and Logistic Regression implementation
│   ├── evaluation.py            # Model metrics and visualizations 
│   └── prediction.py            # Inference engine for single transactions
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── class_distribution.png
│   ├── logistic_model.pkl       # Saved trained model
│   └── scaler.pkl               # Saved fitted scaler
├── main.py                      # Core execution orchestrator
├── requirements.txt
└── README.md
```

## Setup Instructions

**1. Create virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the full pipeline:**
```bash
python main.py
```
This will automatically generate the data if `data/transactions.csv` is missing, run all pipelines, print metrics, and save output charts to the `outputs/` directory.

**3. Test Single Prediction:**
```bash
python src/prediction.py
```

## Results
The model uses `SMOTE` to dramatically improve metric **Recall**, ensuring we identify as many fraudulent transactions as possible. See the `outputs/` folder for the Confusion Matrix and ROC AUC curve confirming the model's reliability in classifying unseen test data.
=======
# FraudShield-ML
>>>>>>> origin/main
