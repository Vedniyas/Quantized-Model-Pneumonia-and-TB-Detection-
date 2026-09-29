# ==============================================================================
# Capstone UI: Chest X-Ray Disease Detection (Using Quantized TFLite Model)
# ==============================================================================
import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------------------------------
# Configuration & Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="X-Ray Diagnostic AI", page_icon="🫁", layout="centered")

# Define target classes matching the training phase
TARGET_CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
IMAGE_SIZE = (224, 224)
MODEL_PATH = "quantized_model.tflite"

# ------------------------------------------------------------------------------
# Model Loading (Cached for fast UI performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}. Error: {e}")
        return None

interpreter = load_tflite_model()

# ------------------------------------------------------------------------------
# Image Processing & Inference Function
# ------------------------------------------------------------------------------
def predict_xray(image, interpreter):
    # 1. Preprocess the image
    img_array = np.array(image)
    
    # Handle Grayscale to RGB conversion if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # Handle PNG with alpha channel
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # 2. Get Input/Output Tensors
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # 3. Run Inference with Latency Tracking
    start_time = time.time()
    
    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    output_probs = interpreter.get_tensor(output_details['index'])[0]
    
    latency_ms = (time.time() - start_time) * 1000

    # 4. Process Results
    predicted_class_idx = np.argmax(output_probs)
    confidence = output_probs[predicted_class_idx] * 100
    predicted_label = TARGET_CLASSES[predicted_class_idx]
    
    return predicted_label, confidence, latency_ms

# ------------------------------------------------------------------------------
# UI Layout & Design
# ------------------------------------------------------------------------------
st.title("🫁 Chest X-Ray Diagnostic AI")
st.markdown("""
This application uses a **Quantized Deep Learning Model (INT8)** to detect abnormalities in Chest X-Rays. 
Upload a scan to get an instant prediction.
""")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    with col2:
        st.markdown("### Analysis Results")
        with st.spinner("Analyzing scan..."):
            if interpreter:
                label, confidence, latency = predict_xray(image, interpreter)
                
                # Display Prediction
                if label == "NORMAL":
                    st.success(f"**Diagnosis:** {label}")
                elif label == "UNKNOWN":
                    st.warning(f"**Diagnosis:** {label} (Requires Radiologist Review)")
                else:
                    st.error(f"**Diagnosis:** {label}")
                
                # Display Metrics
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.metric(label="Inference Latency", value=f"{latency:.2f} ms", help="Time taken by the INT8 model to predict.")
            else:
                st.error("System Error: Model not initialized.")