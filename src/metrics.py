import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Computes standard classification metrics.
    y_true: 1D numpy array of true labels
    y_pred: 1D numpy array of predicted labels
    y_probs: 2D numpy array of predicted class probabilities (optional)
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    
    # Per class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    for i, score in enumerate(per_class_f1):
        metrics[f'f1_class_{i}'] = score
        
    if y_probs is not None:
        metrics['ece'] = calculate_ece(y_probs, y_true)
        
    return metrics

def calculate_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error.
    """
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(preds[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_reliability_diagram(probs, labels, n_bins=10, save_path=None):
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(preds[in_bin] == labels[in_bin]))
        else:
            bin_accs.append(0.0)
            
    ece = calculate_ece(probs, labels, n_bins)
    
    plt.figure(figsize=(6, 6))
    plt.bar(np.arange(n_bins) / n_bins + 0.5 / n_bins, bin_accs, width=1.0/n_bins, edgecolor='black', color='blue', alpha=0.7, label='Outputs')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE = {ece:.4f})')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_classification_report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names)
