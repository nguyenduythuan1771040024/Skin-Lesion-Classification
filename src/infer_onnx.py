import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime as ort

from src.dataset import get_dataloaders, IDX_TO_CLASS
from src.models import get_model

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference Verification")
    parser.add_argument('--config', type=str, default='configs/best_model.yaml', help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    
    # Paths
    pytorch_path = config['save_path']
    onnx_path = pytorch_path.replace('.pth', '.onnx')

    # Load PyTorch model
    pytorch_model = get_model(config['model_name'], num_classes=7, pretrained=False)
    pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=device))
    pytorch_model.eval()

    # Load ONNX Session
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}. Please run export_onnx.py first.")
        return
    ort_session = ort.InferenceSession(onnx_path)

    # Prepare dataloaders
    _, _, test_loader, _, _, test_df = get_dataloaders(
        splits_dir=config['splits_dir'],
        batch_size=1,
        use_weighted_sampler=False
    )

    print("\n=== Comparing PyTorch and ONNX Predictions (First 5 images) ===")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_samples = test_df.head(5)
    for i, (_, row) in enumerate(test_samples.iterrows()):
        img_path = row['image_path']
        img_id = row['image_id']
        true_label = row['dx']
        
        pil_img = Image.open(img_path).convert('RGB')
        tensor_img = transform(pil_img).unsqueeze(0)
        
        # PyTorch
        with torch.no_grad():
            py_out = pytorch_model(tensor_img)
            py_prob = torch.softmax(py_out, dim=1).numpy()[0]
            py_pred = np.argmax(py_prob)
            
        # ONNX
        ort_inputs = {ort_session.get_inputs()[0].name: tensor_img.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        ort_out = torch.tensor(ort_outs[0])
        onnx_prob = torch.softmax(ort_out, dim=1).numpy()[0]
        onnx_pred = np.argmax(onnx_prob)

        print(f"\nImage: {img_id} (True: {true_label.upper()})")
        print(f"  PyTorch -> Pred: {IDX_TO_CLASS[py_pred].upper()} (Prob: {py_prob[py_pred]*100:.2f}%)")
        print(f"  ONNX    -> Pred: {IDX_TO_CLASS[onnx_pred].upper()} (Prob: {onnx_prob[onnx_pred]*100:.2f}%)")
        
        # Verify closeness
        diff = np.max(np.abs(py_prob - onnx_prob))
        print(f"  Max absolute probability difference: {diff:.6e}")
        assert diff < 1e-4, "ONNX and PyTorch outputs are not close enough!"

if __name__ == '__main__':
    main()
