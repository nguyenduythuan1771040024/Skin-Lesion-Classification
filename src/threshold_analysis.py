import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

from src.dataset import get_dataloaders, CLASS_TO_IDX, IDX_TO_CLASS
from src.models import get_model

def plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, thresholds, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Melanoma ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark optimal threshold
    opt_fpr = fpr[optimal_idx]
    opt_tpr = tpr[optimal_idx]
    opt_thresh = thresholds[optimal_idx]
    plt.scatter(opt_fpr, opt_tpr, color='red', s=100, zorder=5, 
                label=f'Optimal Thresh = {opt_thresh:.4f}\n(FPR={opt_fpr:.3f}, TPR={opt_tpr:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve for Melanoma (mel) Detection')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ROC Analysis & Decision Threshold Tuning for mel")
    parser.add_argument('--config', type=str, default='configs/best_model.yaml', help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare dataloaders
    _, _, test_loader, _, _, _ = get_dataloaders(
        splits_dir=config['splits_dir'],
        batch_size=config['batch_size'],
        use_weighted_sampler=False,
        remove_hair=config.get('remove_hair', False)
    )

    # Load Model
    model = get_model(config['model_name'], num_classes=7, pretrained=False)
    if not os.path.exists(config['save_path']):
        print(f"Error: Model checkpoint not found at {config['save_path']}")
        return
        
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    model.to(device)
    model.eval()

    # Extract predictions
    print("Running inference on Test Set...")
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.cpu())
            labels_list.append(labels)
            
    logits = torch.cat(logits_list)
    probs = torch.softmax(logits, dim=1).numpy()
    y_true = torch.cat(labels_list).numpy()

    # Target class 'mel' is index 4
    mel_idx = CLASS_TO_IDX['mel']
    y_true_binary = (y_true == mel_idx).astype(int)
    y_prob_mel = probs[:, mel_idx]

    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_mel)
    roc_auc = auc(fpr, tpr)
    print(f"Melanoma (mel) Test ROC-AUC: {roc_auc:.4f}")

    # Find optimal threshold using Youden's J statistic
    # J = Sensitivity + Specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  at TPR (Sensitivity/Recall): {tpr[optimal_idx]:.4f}")
    print(f"  at FPR (1 - Specificity):    {fpr[optimal_idx]:.4f}")

    # Plot and save ROC curve
    os.makedirs('outputs/figures', exist_ok=True)
    plot_path = 'outputs/figures/roc_curves.png'
    plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, thresholds, plot_path)
    print(f"Saved Melanoma ROC curve to: {plot_path}")

    # Compare classifications
    # 1. Standard prediction (argmax)
    y_pred_std = np.argmax(probs, axis=1)

    # 2. Threshold-adjusted prediction:
    # If prob of mel is >= optimal_threshold, predict mel.
    # Otherwise, predict the argmax of the remaining classes.
    y_pred_adj = []
    for i in range(len(probs)):
        p = probs[i]
        if p[mel_idx] >= optimal_threshold:
            y_pred_adj.append(mel_idx)
        else:
            # Predict the argmax of the other 6 classes
            other_probs = p.copy()
            other_probs[mel_idx] = -1.0 # Suppress target class
            y_pred_adj.append(np.argmax(other_probs))
    y_pred_adj = np.array(y_pred_adj)

    # Class names
    class_names = [IDX_TO_CLASS[i] for i in range(7)]

    print("\n=== CLASSIFICATION REPORT (STANDARD ARGMAX) ===")
    report_std = classification_report(y_true, y_pred_std, target_names=class_names)
    print(report_std)

    print("\n=== CLASSIFICATION REPORT (DECISION THRESHOLD TUNED FOR MEL) ===")
    report_adj = classification_report(y_true, y_pred_adj, target_names=class_names)
    print(report_adj)

    # Save comparison text report
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/threshold_tuning_report.txt', 'w') as f:
        f.write("=== CLASSIFICATION REPORT (STANDARD ARGMAX) ===\n")
        f.write(report_std)
        f.write("\n=== CLASSIFICATION REPORT (DECISION THRESHOLD TUNED FOR MEL) ===\n")
        f.write(report_adj)
        f.write(f"\nTuned Class: Melanoma (mel)\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    print("Saved threshold tuning report to outputs/metrics/threshold_tuning_report.txt")

    # Save threshold json
    import json
    os.makedirs('outputs/models', exist_ok=True)
    with open('outputs/models/threshold.json', 'w') as f:
        json.dump({"optimal_threshold": float(optimal_threshold)}, f)
    print("Saved optimal threshold to outputs/models/threshold.json")


if __name__ == '__main__':
    main()
