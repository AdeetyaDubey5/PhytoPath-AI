import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Get the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.keras")


# Safely load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❗ Failed to load model: {e}")
    st.stop()

# Load the class names
with open(os.path.join(working_dir, "class_indices.json")) as f:
    class_indices = json.load(f)

# Sidebar: show recognized classes, numbered
st.sidebar.title("Recognized Crop Diseases")
formatted_classes = []
for idx, raw_class in class_indices.items():
    if "___" in raw_class:
        fruit, condition = raw_class.split("___")
    else:
        fruit, condition = raw_class, "Unknown"
    fruit = fruit.replace("_", " ").replace("(", "").replace(")", "").title()
    condition = condition.replace("_", " ").replace("(", "").replace(")", "").title()
    formatted_classes.append(f"{fruit} - {condition}")
for i, cls in enumerate(formatted_classes, start=1):
    st.sidebar.markdown(f"{i}. {cls}")

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict and nicely format the result
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    raw_class_name = class_indices[str(predicted_class_index)]
    if "___" in raw_class_name:
        fruit, condition = raw_class_name.split("___")
    else:
        fruit, condition = raw_class_name, "Unknown"
    fruit = fruit.replace("_", " ").replace("(", "").replace(")", "").title()
    condition = condition.replace("_", " ").replace("(", "").replace(")", "").title()
    return fruit, condition

# Streamlit App UI
st.markdown("<h1 style='text-align: center;'>PhytoPath AI</h1>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            fruit, condition = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'**Fruit -** {fruit}\n\n**Condition -** {condition}')
