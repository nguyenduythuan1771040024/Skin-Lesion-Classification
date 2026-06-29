# Skin Lesion Classification using Deep Learning

This project implements a complete, professional Deep Learning pipeline to classify skin lesion images from the HAM10000 dataset into 7 distinct classes.

*Note: For educational and research purposes only. Not for medical diagnosis.*

## Project Structure
```
├── configs/               # YAML configurations for training
│   ├── baseline.yaml      # Baseline model training config
│   ├── model.yaml         # Deep learning transfer model training config
│   └── best_model.yaml    # Config for testing/evaluation
├── data/
│   └── splits/            # CSV dataset train/dev/test splits
├── src/                   # PyTorch modules
│   ├── preprocess.py      # Dataset preparation and split creation
│   ├── train_baseline.py  # Run 1: Baseline CNN
│   ├── train_model.py     # Runs 2-5: Transfer Learning
│   ├── evaluate.py        # Model evaluation script
│   ├── gradcam_visualization.py # Grad-CAM explanation tool
│   ├── export_onnx.py     # Model export to ONNX
│   ├── infer_onnx.py      # ONNX verification script
│   ├── benchmark_inference.py # CPU latency benchmarking
│   └── app.py             # Streamlit interactive application
├── outputs/               # Saved models, confusion matrix, Grad-CAM plots
├── reports/               # Markdown reports and PDF final report
└── requirements.txt
```

## Setup & Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Login to Weights & Biases:
   ```bash
   python -m wandb login <your_api_key>
   ```

## Usage

### 1. Data Splitting & Preprocessing
```bash
python src/preprocess.py
```

### 2. Training
Run Baseline:
```bash
python src/train_baseline.py
```
Run Transfer Learning (EfficientNet):
```bash
python src/train_model.py --config configs/model.yaml --run_name "EfficientNet_FocalLoss"
```

### 3. Evaluation & Grad-CAM
```bash
python src/evaluate.py --config configs/best_model.yaml
python src/gradcam_visualization.py --config configs/best_model.yaml
```

### 4. ONNX Export & Benchmark
```bash
python src/export_onnx.py --config configs/best_model.yaml
python src/infer_onnx.py --config configs/best_model.yaml
python src/benchmark_inference.py --config configs/best_model.yaml
```

### 5. Streamlit App
```bash
streamlit run src/app.py
```

## Results Summary
- **Best Model**: EfficientNet-B0 (Macro-F1: 0.82)
- **ONNX Speedup**: ~2x speedup on CPU
- **Quantization Size Reduction**: ~75% reduction (from 45MB to 11MB)
