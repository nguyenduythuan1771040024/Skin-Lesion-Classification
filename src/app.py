import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import yaml

# Imports from src (when streamlit runs, the root is in Python PATH)
import importlib
import src.dataset
import src.models
import src.gradcam_visualization
importlib.reload(src.dataset)
importlib.reload(src.models)
importlib.reload(src.gradcam_visualization)
from src.dataset import IDX_TO_CLASS, CLASS_TO_IDX, get_transforms
from src.models import get_model
from src.gradcam_visualization import GradCAM, get_target_layer




# Load config
config_path = 'configs/best_model.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {
        'model_name': 'efficientnet',
        'save_path': 'outputs/models/best_model.pth'
    }

st.set_page_config(page_title="Skin Lesion Classification Demo", layout="wide")

st.title("🩺 Deep Learning Skin Lesion Classification")

# Prominent Medical Warning
st.warning("⚠️ **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR MEDICAL DIAGNOSIS.**\n"
           "This system is not a substitute for professional medical advice, diagnosis, or treatment. "
           "Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

@st.cache_resource
def load_model():
    device = torch.device('cpu')  # Force CPU to avoid Windows CUDA multi-threading context errors in Streamlit
    model = get_model(config['model_name'], num_classes=7, pretrained=False)
    if os.path.exists(config['save_path']):
        model.load_state_dict(torch.load(config['save_path'], map_location=device))
    else:
        st.error(f"Model checkpoint not found at {config['save_path']}. Running with random weights.")
    model.to(device)
    model.eval()
    return model, device


try:
    model, device = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

if model is not None:
    _, transform = get_transforms(remove_hair=config.get('remove_hair', False))


    uploaded_file = st.file_uploader("Upload a skin lesion image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            
        tensor_img = transform(image).unsqueeze(0).to(device)
        
        target_layer = get_target_layer(model, config['model_name'])
        gradcam = GradCAM(model, target_layer)
        
        import json
        
        # Load Temperature
        temperature = 1.0
        temp_path = 'outputs/models/temperature.json'
        if os.path.exists(temp_path):
            try:
                with open(temp_path, 'r') as f:
                    temp_data = json.load(f)
                    temperature = temp_data.get('temperature', 1.0)
            except Exception:
                pass
                
        # Load Melanoma Threshold
        optimal_threshold = 0.50
        thresh_path = 'outputs/models/threshold.json'
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, 'r') as f:
                    thresh_data = json.load(f)
                    optimal_threshold = thresh_data.get('optimal_threshold', 0.50)
            except Exception:
                pass

        # Forward pass
        outputs = model(tensor_img)
        
        # Calibrate logits with temperature
        calibrated_outputs = outputs / temperature
        probs = torch.softmax(calibrated_outputs, dim=1).detach().cpu().numpy()[0]
        
        # Standard Prediction (Argmax)
        pred_idx_std = np.argmax(probs)
        pred_class_std = IDX_TO_CLASS[pred_idx_std]
        
        # Threshold-adjusted Prediction for Melanoma (mel)
        mel_idx = CLASS_TO_IDX['mel']
        if probs[mel_idx] >= optimal_threshold:
            pred_idx = mel_idx
        else:
            # Predict the argmax of other classes
            other_probs = probs.copy()
            other_probs[mel_idx] = -1.0
            pred_idx = np.argmax(other_probs)
            
        pred_class = IDX_TO_CLASS[pred_idx]
        
        cam, _ = gradcam(tensor_img, pred_idx)
        gradcam.remove_hooks()
        
        # Apply hair removal to display if enabled
        img_disp = image.copy()
        if config.get('remove_hair', False):
            from src.dataset import HairRemoval
            hair_remover = HairRemoval((224, 224))
            img_disp = hair_remover(img_disp)
            
        img_cv = np.array(img_disp)
        img_cv = cv2.resize(img_cv, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_rgb, 0.4, 0)
        
        with col2:
            st.subheader("Model Predictions")
            
            # Display decision threshold info if adjusted
            if pred_idx != pred_idx_std:
                st.info(f"💡 **Decision Threshold Tuning Active**: Standard argmax prediction was **{pred_class_std.upper()}**, but since the Melanoma probability ({probs[mel_idx]*100:.1f}%) exceeds the clinical safety threshold ({optimal_threshold*100:.1f}%), the system recommends investigating for **MELANOMA**.")
                
            st.metric(label="Predicted Diagnosis", value=pred_class.upper(), delta=f"{probs[pred_idx]*100:.2f}% Confidence (Safety Adjusted)")
            
            if temperature != 1.0:
                st.caption(f"ℹ️ Probabilities calibrated using Temperature Scaling (T = {temperature:.4f})")
                
            prob_dict = {IDX_TO_CLASS[i].upper(): float(probs[i]) for i in range(7)}
            st.bar_chart(prob_dict)

            
        st.subheader("🔍 Explainable AI: Grad-CAM Visualization")
        st.write("Grad-CAM highlights the regions of the image that contributed most to the model's decision.")
        
        cam_col1, cam_col2, cam_col3 = st.columns(3)
        with cam_col1:
            st.image(img_cv, caption="Original (Resized to 224x224)", use_container_width=True)
        with cam_col2:
            st.image(heatmap_rgb, caption="Grad-CAM Heatmap", use_container_width=True)
        with cam_col3:
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
