import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from train_models import CNNMaxPool, CNNAvgPool
from torch.serialization import safe_globals

def calculate_metrics_per_class(conf_matrix):
    """Calculate statistical metrics for each class"""
    n_classes = conf_matrix.shape[0]
    metrics = {
        'PPV': [],  # Positive Predictive Value (Precision)
        'TPR': [],  # True Positive Rate (Recall)
        'TNR': [],  # True Negative Rate (Specificity)
        'NPV': [],  # Negative Predictive Value
        'ACC': [],  # Accuracy
        'DS': []    # Dice Score
    }
    
    for i in range(n_classes):
        # For each class, treat it as the positive class and all others as negative
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - TP - FP - FN
        
        # Calculate metrics
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
        ACC = (TP + TN) / (TP + FP + FN + TN)
        DS = (2 * TP) / ((2 * TP) + FP + FN) if ((2 * TP) + FP + FN) > 0 else 0
        
        metrics['PPV'].append(PPV)
        metrics['TPR'].append(TPR)
        metrics['TNR'].append(TNR)
        metrics['NPV'].append(NPV)
        metrics['ACC'].append(ACC)
        metrics['DS'].append(DS)
    
    return metrics

def plot_confusion_matrix(conf_matrix, title, save_path=None):
    """Plot and optionally save confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_metrics_to_excel(metrics, num_classes, conf_matrix, model_info, save_path='model_metrics.xlsx'):
    """Save metrics and confusion matrix values to Excel file"""
    data = []
    label_mapping = model_info['label_mapping']
    # Reverse the mapping to get model_idx -> original_label
    idx_to_original = {v: k for k, v in label_mapping.items()}
    
    for class_idx in range(num_classes):
        # Calculate TP, TN, FN, FP for this class
        TP = conf_matrix[class_idx, class_idx]
        FP = conf_matrix[:, class_idx].sum() - TP
        FN = conf_matrix[class_idx, :].sum() - TP
        TN = conf_matrix.sum() - TP - FP - FN
        
        row = {
            'Model Class ID': class_idx,
            'Original Class ID': idx_to_original[class_idx],
            'TP': TP,
            'TN': TN,
            'FN': FN,
            'FP': FP,
            'PPV': metrics['PPV'][class_idx],
            'TPR': metrics['TPR'][class_idx],
            'TNR': metrics['TNR'][class_idx],
            'NPV': metrics['NPV'][class_idx],
            'ACC': metrics['ACC'][class_idx],
            'DS': metrics['DS'][class_idx]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add mean values
    means = df.mean(numeric_only=True)
    row = {
        'Model Class ID': 'Mean',
        'Original Class ID': '-',
        'TP': means['TP'],
        'TN': means['TN'],
        'FN': means['FN'],
        'FP': means['FP'],
        'PPV': means['PPV'],
        'TPR': means['TPR'],
        'TNR': means['TNR'],
        'NPV': means['NPV'],
        'ACC': means['ACC'],
        'DS': means['DS']
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save to Excel
    df.to_excel(save_path, index=False)
    print(f"Metrics saved to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the best model
    print("\nLoading best model...")
    state_dict = torch.load('best_model.pth', weights_only=True)
    with safe_globals(['numpy._core.multiarray._reconstruct']):
        model_info = torch.load('model_info.pth', weights_only=False)
    
    # Create appropriate model
    input_shape = model_info['input_shape']
    num_classes = model_info['num_classes']
    model_type = model_info['model_type']
    
    if model_type == 'maxpool':
        model = CNNMaxPool(input_shape, num_classes).to(device)
    else:
        model = CNNAvgPool(input_shape, num_classes).to(device)
    
    model.load_state_dict(state_dict)
    print(f"Loaded {model_type} model")
    
    # Get confusion matrix from model_info (saved during training)
    conf_matrix = model_info['confusion_matrix']
    
    # Calculate metrics
    metrics = calculate_metrics_per_class(conf_matrix)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, f'{model_type} Model Confusion Matrix', 'confusion_matrix.png')
    
    # Save metrics to Excel
    save_metrics_to_excel(metrics, num_classes, conf_matrix, model_info)
    
    # Print mean metrics
    print("\nMean Metrics:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")

if __name__ == "__main__":
    main()