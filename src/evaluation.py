import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os

def evaluate_model(model, X_test, y_test, output_dir='outputs'):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n========= MODEL EVALUATION =========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("====================================\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Saved confusion matrix plot to {output_dir}/confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    print(f"Saved ROC curve plot to {output_dir}/roc_curve.png")
    
    # 3. Save metrics to a text file
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("========= MODEL EVALUATION =========\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write("====================================\n")
        
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def plot_class_distribution(y, output_dir='outputs', title="Class Distribution"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Is Fraud")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_distribution.png'))
    plt.close()
    print(f"Saved class distribution plot to {output_dir}/class_distribution.png")
