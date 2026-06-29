import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import get_dataloaders, IDX_TO_CLASS
from src.models import get_model
from src.trainer import Trainer
from src.metrics import (
    compute_metrics, 
    plot_confusion_matrix, 
    plot_reliability_diagram, 
    get_classification_report
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument('--config', type=str, default='configs/best_model.yaml', help="Path to config yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare dataloaders
    _, _, test_loader, _, _, test_df = get_dataloaders(
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

    # Evaluate
    trainer = Trainer(model=model, optimizer=None, criterion=nn.CrossEntropyLoss(), device=device, save_path=config['save_path'])
    print(f"Evaluating model: {config['model_name']} loaded from {config['save_path']}")
    
    test_metrics, y_true, y_pred, y_probs = trainer.evaluate(test_loader)
    
    class_names = [IDX_TO_CLASS[i] for i in range(7)]

    # 1. Print classification report and save it
    report = get_classification_report(y_true, y_pred, class_names)
    print("\nClassification Report:")
    print(report)
    
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/classification_report.txt', 'w') as f:
        f.write(report)
    print("Saved classification report to outputs/metrics/classification_report.txt")

    # 2. Confusion Matrix
    os.makedirs('outputs/figures', exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names, 'outputs/figures/confusion_matrix.png')
    print("Saved confusion matrix to outputs/figures/confusion_matrix.png")

    # 3. Reliability Diagram
    plot_reliability_diagram(y_probs, y_true, n_bins=10, save_path='outputs/figures/reliability_diagram.png')
    print("Saved reliability diagram to outputs/figures/reliability_diagram.png")

    # 4. Save test predictions CSV
    os.makedirs('outputs/predictions', exist_ok=True)
    image_ids = test_df['image_id'].values
    pred_labels = [IDX_TO_CLASS[pred] for pred in y_pred]
    true_labels = [IDX_TO_CLASS[true] for true in y_true]
    max_probs = np.max(y_probs, axis=1)

    pred_df = pd.DataFrame({
        'image_id': image_ids,
        'true_label': true_labels,
        'predicted_label': pred_labels,
        'probability': max_probs
    })
    pred_df.to_csv('outputs/predictions/test_predictions.csv', index=False)
    print("Saved test predictions to outputs/predictions/test_predictions.csv")

    # 5. Save metrics summary CSV
    summary_df = pd.DataFrame([{
        'Accuracy': test_metrics['accuracy'],
        'Macro-F1': test_metrics['macro_f1'],
        'Recall': test_metrics['recall'],
        'ECE': test_metrics.get('ece', 0.0)
    }])
    summary_df.to_csv('outputs/metrics/metrics_summary.csv', index=False)
    print("Saved metrics summary to outputs/metrics/metrics_summary.csv")

if __name__ == '__main__':
    main()
