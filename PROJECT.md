# Project: Skin Lesion Classification Optimization

## Architecture
- Baseline CNN vs EfficientNet-B0 transfer learning model.
- Preprocessing: Morphological DullRazor hair removal, resizing, normalization.
- Splitting: Group-stratified split based on `lesion_id` to prevent data leakage.
- Training: Merges HAM10000 dataset with scraped supplementary data from ISIC API. Uses WeightedSampler, Focal Loss, class weights.
- Calibration & Tuning: Temperature Scaling to calibrate confidence, Youden's J statistic threshold tuning for Melanoma classification.
- Export: ONNX format (dynamic axis configurations).
- Deployment: Streamlit web application running locally on CPU.
- Reporting: Automated markdown synthesis and PDF generation.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Test Suite Development (E2E Track) | Develop comprehensive opaque-box test suite for features & correctness | None | IN_PROGRESS (Conv: bf333f71-e532-4077-a504-0d8dd74929dc) |
| 2 | Crawl/Scrape Supplementary Data | Scrape ISIC API for df, vasc, akiec (at least 500 images/class in train split) | None | DONE (Conv: 3efb35aa-ee0b-43e7-9a60-996348abdd5c) |
| 3 | Advanced Model Training | Retrain EfficientNet-B0 with group-stratified splits, DullRazor, Focal Loss, weights, WeightedSampler | M2 | IN_PROGRESS (Conv: 935af9cd-ba77-48ae-abfb-2c0f0d26b4f4) |
| 4 | Evaluation & Calibration | Calibrate model, Melanoma threshold tuning, export ONNX, CPU benchmark | M3 | IN_PROGRESS (Conv: 935af9cd-ba77-48ae-abfb-2c0f0d26b4f4) |
| 5 | Streamlit App & Report Sync | Update demo app to support CPU, temperature, threshold; regenerate final PDF | M4, M1 | PLANNED |

## Interface Contracts
- Wrapper scripts at root:
  - `preprocess_skin_images.py`: Runs data preprocessing and splits.
  - `train_transfer_model.py`: Performs fine-tuning.
  - `demo_app.py`: Launches Streamlit dashboard.
- Model Outputs:
  - ONNX Model Path: `outputs/models/best_model.onnx`
  - PyTorch Checkpoint Path: `outputs/models/best_model.pth`
  - Temperature Scaling JSON: `outputs/models/temperature.json` (holds calibrated temperature parameter)
  - Decision Threshold JSON: `outputs/models/threshold.json` (holds decision thresholds for each class, especially melanoma `mel`)
- Data outputs:
  - `data/splits/train.csv`, `data/splits/dev.csv`, `data/splits/test.csv`
  - Scraped images: downloaded to a designated path under `data/` and indexed in metadata csv.
- Metrics outputs:
  - `metrics_summary.csv` and `test_predictions.csv` at root level.
  - `reports/final_report.pdf` inside reports directory.

## Code Layout
- `src/`: Core implementation code.
- `configs/`: Experiment and hyperparameters configuration.
- `data/`: Dataset storage, splits, metadata.
- `outputs/`: Model checkpoints, predictions, figures.
- `reports/`: Markdown and PDF files summarizing performance.
