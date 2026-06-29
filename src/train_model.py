import os
import argparse
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

def freeze_backbone(model, model_name):
    print("Freezing backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False
        
    if 'efficientnet' in model_name.lower():
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif 'densenet' in model_name.lower():
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in list(model.children())[-1].parameters():
            param.requires_grad = True

def unfreeze_all(model):
    print("Unfreezing all parameters for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

def main():
    parser = argparse.ArgumentParser(description="Train Transfer Learning Models")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--run_name', type=str, default=None, help="Name for the W&B run")
    parser.add_argument('--lr_phase1', type=float, default=None, help="Override Phase 1 learning rate")
    parser.add_argument('--lr_phase2', type=float, default=None, help="Override Phase 2 learning rate")
    parser.add_argument('--gamma', type=float, default=2.0, help="Focal Loss gamma parameter")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Overrides (FIXED TYPO HERE)
    if args.lr_phase1 is not None:
        config['lr_phase1'] = args.lr_phase1
    if args.lr_phase2 is not None:
        config['lr_phase2'] = args.lr_phase2

    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Read beta configurations
    sampler_beta = config.get('sampler_beta', 0.35)
    loss_beta = config.get('loss_beta', 0.25)

    # Prepare dataloaders
    train_loader, val_loader, test_loader, train_df, val_df, test_df = get_dataloaders(
        splits_dir=config['splits_dir'],
        batch_size=config['batch_size'],
        use_weighted_sampler=config['use_weighted_sampler'],
        sampler_beta=sampler_beta,
        remove_hair=config.get('remove_hair', False)
    )


    # Initialize Model
    model = get_model(config['model_name'], num_classes=7, pretrained=True)

    # Class Weights if Focal Loss is used
    class_weights = None
    if config['use_focal_loss']:
        # Calculate smooth inverse class frequency weights from train splits using loss_beta
        class_counts = train_df['label_idx'].value_counts().sort_index().values
        class_weights = 1.0 / (class_counts ** loss_beta)
        # Normalize class weights so they average to 1.0
        class_weights = class_weights / np.sum(class_weights) * len(class_counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"Computed smooth class weights (beta={loss_beta}): {class_weights}")
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    # Init W&B
    run_name = args.run_name if args.run_name else f"Run_{config['model_name']}"
    wandb.init(
        project=config['project_name'],
        name=run_name,
        config={**config, **vars(args)}
    )

    # --- PHASE 1: Feature Extraction ---
    print("\n=== PHASE 1: Training Head Only ===")
    freeze_backbone(model, config['model_name'])
    
    lr_phase1 = config['lr_phase1']
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_phase1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        patience=config['patience'],
        save_path=config['save_path']
    )

    trainer.fit(train_loader, val_loader, epochs=config['epochs_phase1'], use_wandb=True)

    # --- PHASE 2: Fine-Tuning ---
    print("\n=== PHASE 2: Fine-Tuning Entire Model ===")
    unfreeze_all(model)
    
    lr_phase2 = config['lr_phase2']
    optimizer = optim.AdamW(model.parameters(), lr=lr_phase2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Re-initialize trainer keeping the best weight path
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        patience=config['patience'],
        save_path=config['save_path']
    )

    trainer.fit(train_loader, val_loader, epochs=config['epochs_phase2'], use_wandb=True)

    # Evaluate on Test
    print("\n=== Final Evaluation on Test Set ===")
    test_metrics, y_true, y_pred, y_probs = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    # Log test metrics
    wandb.log({
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_recall': test_metrics['recall']
    })

    wandb.finish()

if __name__ == '__main__':
    main()
