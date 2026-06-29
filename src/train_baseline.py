import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.dataset import get_dataloaders
from src.models import get_model
from src.trainer import Trainer
from utils.focal_loss import FocalLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Load config
    config_path = 'configs/baseline.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare dataloaders
    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders(
        splits_dir=config['splits_dir'],
        batch_size=config['batch_size'],
        use_weighted_sampler=config['use_weighted_sampler']
    )

    # Model
    model = get_model(config['model_name'], num_classes=7)

    # Loss & Optimizer
    if config['use_focal_loss']:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Init W&B
    wandb.init(
        project=config['project_name'],
        name="Run_1_Baseline_CNN",
        config=config
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        patience=config['patience'],
        save_path=config['save_path']
    )

    print("Starting training of Baseline CNN...")
    trainer.fit(train_loader, val_loader, epochs=config['epochs'], use_wandb=True)

    # Evaluate on Test
    print("Evaluating Baseline CNN on Test Set...")
    test_metrics, y_true, y_pred, y_probs = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    # Log test metrics to W&B
    wandb.log({
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_recall': test_metrics['recall']
    })

    wandb.finish()

if __name__ == '__main__':
    main()
