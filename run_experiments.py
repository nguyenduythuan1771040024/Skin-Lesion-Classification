import subprocess
import shutil
import os

# Set encoding to prevent UnicodeEncodeErrors on Windows consoles when printing emojis
os.environ['PYTHONIOENCODING'] = 'utf-8'

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        result.check_returncode()

def main():
    py_path = r"C:\Users\nguye\.conda\envs\DL\python.exe"
    
    # 0. Data Preparation & Splitting
    # run_cmd(f"{py_path} -m src.preprocess")
    
    # 1. Run 7: Trained with all fixes (Data Leakage fix + DullRazor Hair Removal + 3 Imbalance Methods)
    # run_cmd(f"{py_path} -m src.train_model --config configs/model.yaml --run_name \"Run_7_All_Fixes\"")
    
    # 2. Set up Best Model (copy efficientnet_best.pth to best_model.pth)
    print("\n>>> Setting up Best Model...")
    os.makedirs('outputs/models', exist_ok=True)
    best_src = 'outputs/models/efficientnet_best.pth'
    best_dst = 'outputs/models/best_model.pth'
    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"Copied {best_src} to {best_dst}")
    else:
        print(f"Warning: {best_src} not found.")
                
    # 3. Evaluation
    run_cmd(f"{py_path} -m src.evaluate --config configs/best_model.yaml")
    
    # 4. Probability Calibration (Temperature Scaling)
    run_cmd(f"{py_path} -m src.calibrate_model --config configs/best_model.yaml")
    
    # 5. Threshold Analysis for mel
    run_cmd(f"{py_path} -m src.threshold_analysis --config configs/best_model.yaml")
    
    # 6. Grad-CAM Visualization
    run_cmd(f"{py_path} -m src.gradcam_visualization --config configs/best_model.yaml")
    
    # 7. Export ONNX
    run_cmd(f"{py_path} -m src.export_onnx --config configs/best_model.yaml")
    
    # 8. ONNX Inference Verification
    run_cmd(f"{py_path} -m src.infer_onnx --config configs/best_model.yaml")
    
    # 9. Benchmark ONNX Inference & Quantization
    run_cmd(f"{py_path} -m src.benchmark_inference --config configs/best_model.yaml")
    
    # 10. Generate Reports and PDF
    run_cmd(f"{py_path} -m src.generate_reports")
    
    # 11. Copy files to align with teacher's required paths
    print("\n>>> Copying output files to match teacher's required paths...")
    
    cm_src = 'outputs/figures/confusion_matrix.png'
    cm_dst = 'outputs/confusion_matrix.png'
    if os.path.exists(cm_src):
        shutil.copy(cm_src, cm_dst)
        print(f"Copied {cm_src} to {cm_dst}")
        
    ms_src = 'outputs/metrics/metrics_summary.csv'
    ms_dst = 'metrics_summary.csv'
    if os.path.exists(ms_src):
        shutil.copy(ms_src, ms_dst)
        print(f"Copied {ms_src} to {ms_dst}")
        
    tp_src = 'outputs/predictions/test_predictions.csv'
    tp_dst = 'test_predictions.csv'
    if os.path.exists(tp_src):
        shutil.copy(tp_src, tp_dst)
        print(f"Copied {tp_src} to {tp_dst}")
        
    print("\n=== ALL RUNS AND EVALUATIONS COMPLETED SUCCESSFULLY! ===")

if __name__ == '__main__':
    main()


