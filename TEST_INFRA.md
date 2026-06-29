# E2E Test Infrastructure Documentation

This document describes the opaque-box end-to-end (E2E) testing setup, feature inventory, testing tiers, methodology, and coverage goals for the Skin Lesion Classification Optimization project.

## Opaque-Box E2E Testing Architecture

Opaque-box E2E testing ensures that the entire system functions correctly as a cohesive whole without relying on or exposing implementation internals. The tests invoke wrappers or core entry scripts using CLI commands and evaluate outputs through observable side effects:
- **Command line exit codes**: Verifying that the process exits with `0` (success) or specific non-zero error codes for invalid arguments/boundary cases.
- **Generated output artifacts**: Validating that required model files, metadata tables, figures, reports, and logs are written to their respective paths.
- **Output content checks**: Inspecting CSV and JSON structures (e.g. check that predictions contain valid classes, probability calibrators are within range, and reports are fully generated).

To keep the pipeline testable within seconds on standard CPU environments, a **Mock Data Environment** is initialized using pytest fixtures. This environment builds a tiny subset of the HAM10000 dataset in a temporary folder structure, creating lightweight image files and metadata.

---

## Features Inventory

The test suite validates the following 8 core system features (F1 - F8):

| ID | Feature Name | Description |
|---|---|---|
| **F1** | Image Preprocessing & Hair Removal | Morphological DullRazor hair removal (blackhat filter, thresholding, inpainting), resizing to 224x224, and color channel normalization. |
| **F2** | Data Splitting & Stratification | Group-stratified split by `lesion_id` to prevent data leakage (same patient/lesion in both train and validation/test splits) while maintaining class ratios. |
| **F3** | Deep Learning Model Training | Training wrapper supporting baseline CNN and Transfer Learning backbones (EfficientNet-B0/DenseNet121) with config overrides, Focal Loss, class weighting, and WeightedRandomSampler. |
| **F4** | Model Probability Calibration | Post-training Temperature Scaling to calibrate confidence scores, verified via Expected Calibration Error (ECE) calculations. |
| **F5** | Decision Threshold Tuning | Tuning classification thresholds specifically for Melanoma (`mel`) using Youden's J statistic from ROC analysis to balance sensitivity and specificity. |
| **F6** | ONNX Export & Benchmarking | Exporting trained PyTorch models to ONNX formats with dynamic batch sizes, verifying probability equivalence, and benchmarking CPU inference speeds against FP32 and INT8 quantized models. |
| **F7** | Automated Report Generation | Markdown error analyses, deployment notes, compression reports, and final PDF generation using ReportLab. |
| **F8** | Streamlit Web Interface | Interactive app displaying predictions, explainable Grad-CAM heatmaps, safety-adjusted thresholds, and calibrated confidence levels. |

---

## Testing Methodology & Coverage Goals

Our verification strategy divides the test cases into 4 distinct tiers:

### 1. Tier 1: Feature Coverage (>= 5 test cases per feature, total >= 30)
Verifies the functional correctness of every feature under standard, happy-path conditions.
- **F1**: Checks each stage of DullRazor (grayscale, morphology, inpainting), resizing, and normalization.
- **F2**: Verifies stratified splitting, no lesion leakage, split files creation, ratio preservation, and grouping.
- **F3**: Checks baseline model initialization, training overrides, sampler activation, focal loss, and early stopping.
- **F4**: Validates temperature scaling optimization, ECE computation, JSON export, reliability plot, and probability sum.
- **F5**: Checks J statistic tuning, threshold JSON output, ROC curves, recall increase, and report contents.
- **F6**: Validates ONNX export, output closeness (diff < 1e-4), dynamic axes, speed benchmarks, and comparison CSV.
- **F7**: Validates error markdown, deployment markdown, compression markdown, experiment logs, and final PDF.
- **F8**: Checks streamlit configs, prediction routing, Grad-CAM overlays, slider ranges, and page routes.

### 2. Tier 2: Boundary & Corner Cases (>= 5 test cases per feature, total >= 30)
Evaluates system robustness under extreme, unusual, or degenerate inputs.
- **F1**: Handles empty (zero bytes), all-black, all-white, non-standard ratio, and ultra-high resolution images.
- **F2**: Handles single-sample classes, severe imbalances, empty metadata, duplicate IDs, and missing files.
- **F3**: Evaluates training with batch size > dataset, zero epochs, extreme learning rates, invalid parameters, and CPU fallback.
- **F4**: Verifies calibration with pre-calibrated logits, zero/flat logits, extreme values, single sample, and unseen classes.
- **F5**: Checks threshold tuning with zero/one probabilities, no target cases, only target cases, and NaN inputs.
- **F6**: Validates ONNX overwrite protection, zero sample benchmark, empty batch, missing ONNX quantization, and massive batch size.
- **F7**: Assesses report writing with missing inputs, zero errors, all errors, missing ReportLab library, and special character injection.
- **F8**: Evaluates app with missing uploads, unsupported file types, offline modes, missing checkpoints, and missing thresholds.

### 3. Tier 3: Cross-Feature Combinations (total >= 6)
Ensures smooth data handoffs and system integration between multiple features.
- Preprocessing outputs directly feed into Splitting.
- Split files are read and processed by the Model Training loaders.
- Trained checkpoints feed into Probability Calibration.
- Calibration JSON feeds into Threshold Tuning.
- Tuned thresholds influence ONNX inference.
- ONNX performance metrics directly feed into automated markdown/PDF reports.

### 4. Tier 4: Real-World Application Scenarios (total >= 5)
Simulates end-to-end workflows of actors interacting with the system.
- **Scenario A**: Developer running the raw data ingestion pipeline to produce a production-ready ONNX model.
- **Scenario B**: Researcher fine-tuning, calibrating, and exporting performance logs for clinical validation.
- **Scenario C**: Clinician uploading a skin lesion to the Streamlit app and receiving safety-adjusted diagnostics.
- **Scenario D**: Clinician exporting a PDF diagnostic report after checking Grad-CAM visual cues.
- **Scenario E**: MLOps engineer dynamically quantizing models for low-power edge deployment and auditing footprint sizes.

---

## Test Execution

All E2E tests are run in a CPU-only environment using the python interpreter at `C:\Users\nguye\.conda\envs\DL\python.exe`.

Run the entire suite from the project root using:
```bash
C:\Users\nguye\.conda\envs\DL\python.exe -m pytest -v tests/test_e2e.py
```
