import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from camera_input_live import camera_input_live
import cv2
import matplotlib.pyplot as plt
import convert_to_tfdataset
from streamlit_option_menu import option_menu


selected = option_menu("Main Menu", ["Video Mode", 'Camera Mode', 'Test Mode'], default_index=0, orientation="horizontal")

st.title("Operator Recognition")


model = tf.keras.models.load_model('model.h5')


def process_image(image):
    img = Image.open(image)
    img_array = np.array(img)
    return img_array

def preprocessor(image):
    # change image array to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize image but don't stitch it, don't keep aspect ratio
    # image = cv2.resize(image, (84, 28), interpolation=cv2.INTER_AREA)
    image = image[:, 28:112]
    # enhance the contrast of the image
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # image = clahe.apply(image)
    # normalize image
    image = image / 255.0
    # reshape image (my model is trained on 28x84x1)
    image = image.reshape(1, 28, 84, 1)
    return image

if selected == "Video Mode":
    image = camera_input_live()
    st.image(image)
    image = process_image(image)
    image = preprocessor(image)
    st.image(image)
    st.subheader(model.predict(image, verbose=0)[0][0])

if selected == "Camera Mode":
    image = st.camera_input("Camera")
    if image:
        image = process_image(image)
        image = preprocessor(image)
        st.image(image)
        st.subheader(model.predict(image, verbose=0)[0][0])

if selected == "Test Mode":
    X_train, y_train, X_test, y_test = convert_to_tfdataset.with_operator_dataset(k=1)
    st.button("Generate New Image")
    image = X_test[0].reshape(1, 28, 84, 1)
    st.image(image)
    st.subheader(model.predict(image, verbose=0)[0][0])
