import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.set_page_config(
    page_title="REAL vs FAKE FACES PREDICTOR APP",
    page_icon="🎭",
)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://orthodoxreflections.com/wp-content/uploads/2023/02/mass-surveillance.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
# Load the trained model
loaded_model = load_model("trained_model.h5")

new_title = '<p style="font-family:sans-serif; color:white; font-size: 35px;">REAL VS FAKE FACES PREDICTOR  APP</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Function to preprocess the input image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(96, 96))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocess the user input image
    st.image(uploaded_file)
    generate_pred = st.button("Generate Prediction")
    if generate_pred:
        # Preprocess the uploaded image
        processed_image = preprocess_image(uploaded_file)
        # Perform prediction using the loaded model
        prediction = loaded_model.predict(processed_image)
        # Determine the predicted class (e.g., REAL or FAKE)
        predicted_class = "REAL" if prediction[0][0] >= 0.5 else "FAKE"
        st.write(f"<span style='color: white; font-size: 30px;'>Predicted: {predicted_class}</span>", unsafe_allow_html=True)

