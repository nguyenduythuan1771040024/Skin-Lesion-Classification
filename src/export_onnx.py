import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import torch
from src.models import get_model

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument('--config', type=str, default='configs/best_model.yaml', help="Path to config yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cpu') # Exporting on CPU is standard and avoids cuda dependencies in ONNX
    
    # Instantiate model
    model = get_model(config['model_name'], num_classes=7, pretrained=False)
    
    # Load weights
    if not os.path.exists(config['save_path']):
        print(f"Error: Model checkpoint not found at {config['save_path']}")
        return
        
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    model.eval()
    
    # Export path
    onnx_path = config['save_path'].replace('.pth', '.onnx')
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("ONNX Export complete.")

if __name__ == '__main__':
    main()
