import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.models import get_model
from src.metrics import calculate_ece

def plot_side_by_side_reliability(probs_before, probs_after, labels, save_path, n_bins=10):
    labels = np.array(labels)
    
    def get_bin_accs(probs):
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
        return bin_accs

    ece_before = calculate_ece(probs_before, labels, n_bins)
    ece_after = calculate_ece(probs_after, labels, n_bins)
    
    bin_accs_before = get_bin_accs(probs_before)
    bin_accs_after = get_bin_accs(probs_after)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Before Calibration
    axes[0].bar(np.arange(n_bins) / n_bins + 0.5 / n_bins, bin_accs_before, width=1.0/n_bins, edgecolor='black', color='red', alpha=0.6, label='Outputs')
    axes[0].plot([0, 1], [0, 1], color='blue', linestyle='--', label='Perfect Calibration')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Uncalibrated (ECE = {ece_before:.4f})')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # After Calibration
    axes[1].bar(np.arange(n_bins) / n_bins + 0.5 / n_bins, bin_accs_after, width=1.0/n_bins, edgecolor='black', color='green', alpha=0.6, label='Outputs')
    axes[1].plot([0, 1], [0, 1], color='blue', linestyle='--', label='Perfect Calibration')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Calibrated with Temperature (ECE = {ece_after:.4f})')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Probability Calibration (Temperature Scaling) Comparison", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Model Probability Calibration")
    parser.add_argument('--config', type=str, default='configs/best_model.yaml', help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare dataloaders
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        splits_dir=config['splits_dir'],
        batch_size=config['batch_size'],
        use_weighted_sampler=False,
        remove_hair=config.get('remove_hair', False)
    )

    # Load Model
    model = get_model(config['model_name'], num_classes=7, pretrained=False)
    if not os.path.exists(config['save_path']):
        print(f"Error: Model file {config['save_path']} not found.")
        return
        
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    model.to(device)
    model.eval()

    print("Extracting validation (dev) logits...")
    val_logits_list = []
    val_labels_list = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_logits_list.append(outputs.cpu())
            val_labels_list.append(labels)
            
    val_logits = torch.cat(val_logits_list)
    val_labels = torch.cat(val_labels_list)

    # Optimize temperature
    print("Optimizing temperature scaling parameter...")
    temperature = torch.ones(1, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    
    # We use LBFGS which is standard for temperature scaling
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(val_logits / temperature, val_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    
    optimal_temp = temperature.item()
    print(f"Optimal Temperature (T): {optimal_temp:.4f}")

    # Extract test set logits for evaluation
    print("Extracting test logits...")
    test_logits_list = []
    test_labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_logits_list.append(outputs.cpu())
            test_labels_list.append(labels)
            
    test_logits = torch.cat(test_logits_list)
    test_labels = torch.cat(test_labels_list)

    # Calculate probabilities
    probs_before = torch.softmax(test_logits, dim=1).numpy()
    probs_after = torch.softmax(test_logits / optimal_temp, dim=1).numpy()
    test_labels_np = test_labels.numpy()

    ece_before = calculate_ece(probs_before, test_labels_np)
    ece_after = calculate_ece(probs_after, test_labels_np)
    
    print(f"Test ECE Before Calibration: {ece_before:.4f}")
    print(f"Test ECE After Calibration:  {ece_after:.4f}")

    # Save Reliability diagram comparison
    os.makedirs('outputs/figures', exist_ok=True)
    plot_path = 'outputs/figures/reliability_diagram.png'
    plot_side_by_side_reliability(probs_before, probs_after, test_labels_np, plot_path)
    print(f"Saved calibrated reliability diagram comparison to: {plot_path}")

    # Save temperature parameter
    os.makedirs('outputs/models', exist_ok=True)
    temp_path = 'outputs/models/temperature.json'
    with open(temp_path, 'w') as f:
        json.dump({"temperature": optimal_temp}, f)
    print(f"Saved temperature parameter to: {temp_path}")

if __name__ == '__main__':
    main()
