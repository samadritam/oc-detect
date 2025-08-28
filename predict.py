import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

model = tf.keras.models.load_model("oral_cancer_cnn.h5")

class_labels = ["CANCER", "NON-CANCER"]

def predict_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    if 0.3 <= confidence <= 0.7:
        label = "DOUBTFUL"
    else:
        label = class_labels[predicted_class]

    return label, confidence

st.title("🦷 Oral Cancer Detection 🦷")
st.write("Upload an oral cavity image, and the model will predict if it is **Non-Cancer**, **Cancer**, or **Doubtful**.")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    label, confidence = predict_image(img)

    st.markdown(f"### 🔎 Prediction: **{label}**")
    #st.write(f"✅ Confidence: {confidence:.4f}")




