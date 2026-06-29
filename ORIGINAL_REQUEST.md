# Original User Request

## Initial Request — 2026-06-28T23:56:41+07:00

Optimize the skin lesion classification deep learning project (EfficientNet-B0 on HAM10000) to minimize misclassifications, supplement underrepresented classes by downloading images from the ISIC Archive API, and maintain strict compliance with academic and deployment requirements.

Working directory: `c:/DL/Project`
Integrity mode: development

## Requirements

### R1. Crawl/Scrape Supplementary Data from ISIC Archive API
Download additional high-quality skin lesion images for the minority classes (Dermatofibroma `df`, Vascular lesions `vasc`, and Actinic keratoses `akiec`) from the official ISIC Archive API (https://api.isic-archive.com/v2/). Supplement each of these classes to have at least 500 images in the training dataset split. Ensure metadata mapping is consistent.

### R2. Advanced Model Fine-Tuning & Training
Merge the supplemented data into the pipeline and retrain the model. Use the existing preprocessing pipeline:
- No data leakage: Ensure group-stratified splitting based on `lesion_id` is maintained so that no lesion's images appear in both train and validation/test splits.
- Morphological DullRazor hair removal preprocessing must be applied to all images.
- Imbalance handling: Apply focal loss, class weighting, and weighted sampler.

### R3. Evaluation and Deployment Calibration
- Evaluate the retrained model on the test split.
- Run Temperature Scaling to calibrate prediction probabilities and generate reliability diagrams.
- Perform Melanoma threshold tuning using Youden's J statistic.
- Export the updated model to ONNX format (ensuring dynamic axis configurations).
- Run CPU inference latency benchmarks.

### R4. Streamlit App & Report Synchronization
- Update the Streamlit application to load the newly trained model, optimal temperature, and optimal Melanoma threshold.
- Keep all wrapper scripts at the root level (`preprocess_skin_images.py`, `train_transfer_model.py`, `demo_app.py`) fully working.
- Re-generate the final PDF report (`reports/final_report.pdf`) with updated metrics, comparison tables, and figures.

## Acceptance Criteria

### Data & Correctness
- [ ] At least 500 images are present in the training split for `df`, `vasc`, and `akiec`.
- [ ] Zero data leakage: `lesion_id` groups are strictly isolated between train, validation, and test splits.
- [ ] Average Macro-F1 score on the Test set increases compared to the baseline (current: 62.98%).
- [ ] Melanoma (`mel`) Recall remains at 80% or higher, with probability calibration ECE below 5%.

### Executables & Artifacts
- [ ] Root wrapper scripts `preprocess_skin_images.py`, `train_transfer_model.py`, and `demo_app.py` execute without error.
- [ ] Updated Streamlit app runs on CPU and dynamically performs predictions with hair removal, temperature calibration, and tuned decision threshold warning.
- [ ] `final_report.pdf` is successfully generated and contains the new experimental results.
