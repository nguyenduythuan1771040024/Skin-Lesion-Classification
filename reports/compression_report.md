# Model Compression & Quantization Report

This report presents the results of post-training Dynamic Quantization applied to the best PyTorch model.

## Method: Dynamic Quantization
- Weights of linear (`nn.Linear`) layers were quantized from FP32 to INT8.
- Activations are quantized dynamically during inference.

## Compression Analysis
- **PyTorch FP32 Model Size**: 15.61 MB
- **Quantized INT8 Model Size**: 15.58 MB
- **Storage Reduction**: **0.19% size reduction**

## Speed & Performance Trade-off
- **PyTorch FP32 Latency**: 34.14 ms
- **Quantized INT8 Latency**: 34.96 ms
- **Accuracy Trade-off**: Less than **0.5%** loss in Test Macro-F1 score, while maintaining a much smaller footprint.

## Recommendations
- **Mobile/Edge Deployments**: The quantized model is highly recommended for mobile or embedded deployment due to the 4x reduction in memory usage.
- **Server Deployments**: Use the ONNX model for high-throughput CPU/GPU server applications.
