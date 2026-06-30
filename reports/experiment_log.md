# Experiment Log & Summary

This document logs all training runs conducted during the development of the skin lesion classification project.

## Experiment Summary Table

| Run   | Model                   | Key Setting                                      |   Validation Metric (F1) |   Test Metric (F1) |
|:------|:------------------------|:-------------------------------------------------|-------------------------:|-------------------:|
| Run 1 | Baseline CNN            | Weighted Sampler, CE Loss                        |                    0.45  |              0.43  |
| Run 2 | EfficientNet-B0         | Phase 1 & 2, CE Loss                             |                    0.78  |              0.76  |
| Run 3 | EfficientNet-B0 (Tuned) | Phase 1 & 2, Focal Loss (gamma=3)                |                    0.82  |              0.81  |
| Run 4 | DenseNet121 (Baseline)  | Phase 1 & 2, Focal Loss (gamma=2)                |                    0.8   |              0.79  |
| Run 5 | Best Model (Old)        | Tuned hyperparams, Focal Loss + Inv Freq Weights |                    0.83  |              0.82  |
| Run 6 | EfficientNet-B0         | Data Leakage (Image-level split)                 |                    0.786 |              0.803 |
| Run 7 | EfficientNet-B0         | Leak-free split, DullRazor, Class weights        |                    0.647 |              0.63  |
| Run 8 | EfficientNet-B0 (Final) | ISIC Augmented, Focal Loss fixed, DullRazor      |                    0.711 |              0.71  |
| Run 9 | DenseNet121 (Final)     | ISIC Augmented, Focal Loss fixed, DullRazor      |                    0.706 |              0.721 |

## Detailed Run Notes

### Run 1: Baseline CNN
- **Model**: BaselineCNN
- **Dataset split**: Stratified (70/15/15)
- **Key hyperparameters**: lr=1e-3, batch_size=32, WeightedRandomSampler
- **Validation result**: F1 = 0.43
- **Test result**: F1 = 0.41
- **Nhận xét**: Overfits quickly on the majority class (`nv`) despite WeightedRandomSampler. Shows poor generalization on rare classes (e.g. `df`, `vasc`).

### Run 2: EfficientNet-B0 (Standard)
- **Model**: EfficientNet-B0
- **Dataset split**: Stratified (70/15/15)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=1e-4, CE Loss
- **Validation result**: F1 = 0.78
- **Test result**: F1 = 0.76
- **Nhận xét**: Drastic improvement in Macro-F1 due to deep pretrained representations.

### Run 3: EfficientNet-B0 (Tuned)
- **Model**: EfficientNet-B0
- **Dataset split**: Stratified (70/15/15)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=5e-5, Focal Loss (gamma=3)
- **Validation result**: F1 = 0.82
- **Test result**: F1 = 0.81
- **Nhận xét**: Replaced CrossEntropyLoss with Focal Loss ($\gamma=3.0$) and decreased fine-tuning learning rate. Improved recall for minority classes (`akiec`, `mel`), raising Macro-F1.

### Run 4: DenseNet121
- **Model**: DenseNet121
- **Dataset split**: Stratified (70/15/15)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=1e-4, Focal Loss (gamma=2)
- **Validation result**: F1 = 0.80
- **Test result**: F1 = 0.79
- **Nhận xét**: Comparable performance to EfficientNet-B0, but model size is larger and inference time is slightly slower.

### Run 5: Best Model (Old)
- **Model**: EfficientNet-B0
- **Dataset split**: Stratified (70/15/15)
- **Key hyperparameters**: Optimized hyperparams, Focal Loss + Inverse Frequency weights
- **Validation result**: F1 = 0.83
- **Test result**: F1 = 0.82
- **Nhận xét**: Best trade-off between model size, inference speed, and classification metrics before data leakage check.

### Run 6: EfficientNet-B0 (Standard Split)
- **Model**: EfficientNet-B0
- **Dataset split**: Image-level Stratified Split (Data Leakage)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=1e-4, Focal Loss + Sampler + DullRazor
- **Validation result**: F1 = 0.786
- **Test result**: F1 = 0.803
- **Nhận xét**: Artificially inflated metrics due to data leakage (same patient lesions present in both train and test splits).

### Run 7: EfficientNet-B0 (Leak-free split)
- **Model**: EfficientNet-B0
- **Dataset split**: Group-Stratified Split by lesion_id (No Leakage)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=1e-4, Focal Loss + Sampler + DullRazor
- **Validation result**: F1 = 0.647
- **Test result**: F1 = 0.630
- **Nhận xét**: True baseline after correcting data leakage. F1 drop is expected and reflects honest generalization ability.

### Run 8: EfficientNet-B0 (Final)
- **Model**: EfficientNet-B0
- **Dataset split**: Group-Stratified Split by lesion_id + ISIC Augmented (No Leakage)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=1e-4, Focal Loss (fixed alpha weight) + Sampler + DullRazor
- **Validation result**: F1 = 0.711
- **Test result**: F1 = 0.710
- **Nhận xét**: Drastic improvement (+8.0% F1) achieved by balancing minority classes with ISIC data and correcting Focal Loss class weights.

### Run 9: DenseNet121 (Comparison)
- **Model**: DenseNet121
- **Dataset split**: Group-Stratified Split by lesion_id + ISIC Augmented (No Leakage)
- **Key hyperparameters**: lr_phase1=1e-3, lr_phase2=1e-4, Focal Loss + Sampler + DullRazor
- **Validation result**: F1 = 0.706
- **Test result**: F1 = 0.721
- **Nhận xét**: Achieves slightly higher F1-score (+1.1%) than EfficientNet-B0, but requires double the parameters and runs slower on CPU.
