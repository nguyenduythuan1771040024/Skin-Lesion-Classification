import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import yaml
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from src.dataset import get_dataloaders
from src.models import get_model

def get_file_size_mb(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="ONNX and Quantization Benchmark")
    parser.add_argument('--config', type=str, default='configs/best_model.yaml', help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu') # Benchmark on CPU for edge/server comparison
    print("Running speed benchmarks on CPU...")

    # Paths
    pytorch_path = config['save_path']
    onnx_path = pytorch_path.replace('.pth', '.onnx')
    quantized_path = pytorch_path.replace('.pth', '_quantized.pth')

    # Load PyTorch model
    pytorch_model = get_model(config['model_name'], num_classes=7, pretrained=False)
    pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=device))
    pytorch_model.eval()

    # Load/Create Quantized model
    if os.path.exists(quantized_path):
        quantized_model = torch.quantization.quantize_dynamic(
            pytorch_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized_model.load_state_dict(torch.load(quantized_path, map_location=device))
    else:
        print("Creating dynamically quantized model...")
        quantized_model = torch.quantization.quantize_dynamic(
            pytorch_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), quantized_path)
    quantized_model.eval()

    # Load ONNX Session
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}. Please run export_onnx.py first.")
        return
    ort_session = ort.InferenceSession(onnx_path)

    # Prepare dataloaders
    _, _, test_loader, _, _, _ = get_dataloaders(
        splits_dir=config['splits_dir'],
        batch_size=1,
        use_weighted_sampler=False,
        remove_hair=config.get('remove_hair', False)
    )


    print("\n=== Benchmarking Inference Latency (20 samples) ===")
    
    py_times = []
    onnx_times = []
    q_times = []
    
    count = 0
    for images, labels in test_loader:
        if count >= 20:
            break
            
        # Benchmark PyTorch
        t0 = time.time()
        with torch.no_grad():
            _ = pytorch_model(images)
        py_times.append(time.time() - t0)
        
        # Benchmark ONNX
        t0 = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
        _ = ort_session.run(None, ort_inputs)
        onnx_times.append(time.time() - t0)
        
        # Benchmark Quantized
        t0 = time.time()
        with torch.no_grad():
            _ = quantized_model(images)
        q_times.append(time.time() - t0)
        
        count += 1

    avg_py = np.mean(py_times) * 1000
    avg_onnx = np.mean(onnx_times) * 1000
    avg_q = np.mean(q_times) * 1000

    print(f"Avg Latency (CPU):")
    print(f"  PyTorch FP32: {avg_py:.2f} ms")
    print(f"  ONNX CPU: {avg_onnx:.2f} ms")
    print(f"  Quantized INT8: {avg_q:.2f} ms")

    # Save sizes
    size_py = get_file_size_mb(pytorch_path)
    size_onnx = get_file_size_mb(onnx_path)
    size_q = get_file_size_mb(quantized_path)
    
    comp_data = [
        {"Model": "PyTorch", "Size (MB)": f"{size_py:.2f}", "Avg Inference Time (ms)": f"{avg_py:.2f}"},
        {"Model": "ONNX", "Size (MB)": f"{size_onnx:.2f}", "Avg Inference Time (ms)": f"{avg_onnx:.2f}"},
        {"Model": "Quantized", "Size (MB)": f"{size_q:.2f}", "Avg Inference Time (ms)": f"{avg_q:.2f}"}
    ]
    
    comp_df = pd.DataFrame(comp_data)
    os.makedirs('reports', exist_ok=True)
    comp_df.to_csv('reports/model_comparison.csv', index=False)
    print("\nSaved comparison to reports/model_comparison.csv")
    print(comp_df.to_markdown(index=False))

if __name__ == '__main__':
    main()
