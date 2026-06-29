import os
import torch
import numpy as np
from tqdm import tqdm
import wandb
from src.metrics import compute_metrics

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, device='cuda', patience=5, save_path='best_model.pth'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.save_path = save_path
        
        self.best_metric = -1.0
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = epoch_loss
        return metrics

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation/Test", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                
        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics['loss'] = epoch_loss
        return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def fit(self, train_loader, val_loader, epochs=10, use_wandb=False):
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics, _, _, _ = self.evaluate(val_loader)
            
            val_loss = val_metrics['loss']
            val_macro_f1 = val_metrics['macro_f1']
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val F1: {val_macro_f1:.4f}")
            
            # Log to Wandb
            if use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_macro_f1': train_metrics['macro_f1'],
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_macro_f1': val_macro_f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
                
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Early stopping check based on val_macro_f1 (maximizing)
            if val_macro_f1 > self.best_metric:
                self.best_metric = val_macro_f1
                self.patience_counter = 0
                # Save model
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.save_path)
                print(f"--> Saved best model with Val F1: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    break
        
        # Load best model weight before returning
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path))
            print("Loaded best weights for final evaluation.")
