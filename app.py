import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import tensorflow.lite as tflite  # used inside interpreter

# MUST be the first Streamlit command
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# ==========================
# Load and cache TFLite model
# ==========================
@st.cache_resource
def load_tflite_model():
    model_path = "brain_tumor_model.tflite"

    # Download model if not available
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading TFLite model..."):
            url = "https://drive.google.com/uc?id=1IVSNk-_apRYtiS32Lh-ZHBB7JvwmWUun"
            gdown.download(url, model_path, quiet=False)

    # Load TFLite model into interpreter
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['No Tumor', 'Pituitary Tumor']

# ==========================
# Theme + Sidebar
# ==========================
if 'history' not in st.session_state:
    st.session_state.history = []

theme_mode = st.sidebar.radio("🌗 Theme", ("Light", "Dark"))

if theme_mode == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("### 🤖 AI Assistant")
user_input = st.sidebar.text_input("Ask me anything")

if user_input:
    st.sidebar.markdown("AI Response:")
    if "tumor" in user_input.lower():
        st.sidebar.write("Tumors are abnormal cell growths. This tool helps detect Pituitary tumors using MRI scans.")
    elif "accuracy" in user_input.lower():
        st.sidebar.write("The model outputs predictions with confidence, but always consult a medical expert.")
    else:
        st.sidebar.write("I'm here to help you understand tumor predictions!")

# ==========================
# Title and Description
# ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #6a0dad;'>🧠 Brain Tumor Classification</h1>
    <p style='text-align: center;'>Upload an MRI image to detect brain tumors using AI</p>
    """,
    unsafe_allow_html=True
)

# ==========================
# File Uploader
# ==========================
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown("### 🖼 Uploaded Image")
    st.image(image, use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    confidence = output[0][predicted_class]

    # Display result
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        if predicted_class == 0:
            st.success("✅ No Tumor Detected")
        else:
            st.error("⚠ Pituitary Tumor Detected")

    with col2:
        st.markdown("Confidence Level:")
        st.progress(int(confidence * 100))
        st.write(f"{confidence * 100:.2f}%")

    # Save to history
    st.session_state.history.append({
        "image": uploaded_file.name,
        "prediction": classes[predicted_class],
        "confidence": f"{confidence * 100:.2f}%"
    })

# ==========================
# Prediction History
# ==========================
with st.expander("🕓 View Prediction History"):
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"{i}. *{entry['image']}* — Prediction: {entry['prediction']}, Confidence: {entry['confidence']}")
    else:
        st.write("No history yet.")

st.markdown("---")
st.caption("🔍 Note: This tool is for educational/demo purposes only. Always consult a specialist for medical advice.")
