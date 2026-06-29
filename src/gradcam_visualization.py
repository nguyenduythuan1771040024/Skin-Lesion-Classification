import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import get_dataloaders, IDX_TO_CLASS, get_transforms
from src.models import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        try:
            self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        except AttributeError:
            self.backward_hook = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def get_target_layer(model, model_name):
    if model_name.lower() == 'baseline':
        return model.features[8]
    elif model_name.lower() == 'efficientnet':
        return model.features[-1][0]
    elif model_name.lower() == 'densenet':
        return model.features.norm5
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations")
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
        batch_size=1,
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

    # Get target layer
    target_layer = get_target_layer(model, config['model_name'])
    gradcam = GradCAM(model, target_layer)

    correct_samples = []
    wrong_samples = []
    
    print("Running inference to find correct and incorrect samples...")
    idx = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            true = labels.cpu().item()
            
        row = test_df.iloc[idx]
        sample_info = {
            'image_path': row['image_path'],
            'image_id': row['image_id'],
            'true_idx': true,
            'pred_idx': pred,
            'prob': probs[pred]
        }
        
        if pred == true and len(correct_samples) < 5:
            correct_samples.append(sample_info)
        elif pred != true and len(wrong_samples) < 5:
            wrong_samples.append(sample_info)
            
        if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
            break
        idx += 1

    # Prepare output directories under outputs/figures/
    os.makedirs('outputs/figures/gradcam_correct', exist_ok=True)
    os.makedirs('outputs/figures/gradcam_wrong', exist_ok=True)
    os.makedirs('outputs/figures/error_samples', exist_ok=True)

    _, model_transform = get_transforms(remove_hair=config.get('remove_hair', False))

    def process_and_save(samples, save_dir):
        for i, s in enumerate(samples):
            img_path = s['image_path']
            image_id = s['image_id']
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            pil_img = Image.open(img_path).convert('RGB')
            tensor_img = model_transform(pil_img).unsqueeze(0).to(device)
            
            cam, _ = gradcam(tensor_img, s['true_idx'])
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            overlay = cv2.addWeighted(img_resized, 0.6, heatmap_rgb, 0.4, 0)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_resized)
            axes[0].set_title(f"Original: {image_id}\nTrue: {IDX_TO_CLASS[s['true_idx']]}")
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_rgb)
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nPred: {IDX_TO_CLASS[s['pred_idx']]} ({s['prob']*100:.1f}%)")
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{image_id}_gradcam.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved Grad-CAM to {save_path}")
            
            if save_dir == 'outputs/figures/gradcam_wrong':
                error_save_path = os.path.join('outputs/figures/error_samples', f"{image_id}_error.png")
                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.title(f"True: {IDX_TO_CLASS[s['true_idx']]} | Pred: {IDX_TO_CLASS[s['pred_idx']]} ({s['prob']*100:.1f}%)")
                plt.axis('off')
                plt.savefig(error_save_path)
                plt.close()

    print("\nProcessing correctly predicted samples...")
    process_and_save(correct_samples, 'outputs/figures/gradcam_correct')
    
    print("\nProcessing incorrectly predicted samples...")
    process_and_save(wrong_samples, 'outputs/figures/gradcam_wrong')
    
    gradcam.remove_hooks()
    print("\nGrad-CAM generation complete.")

if __name__ == '__main__':
    main()
