import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Set page config
st.set_page_config(page_title="Pixelate Image App")

# App title
st.title("Pixelate Your Image")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using PIL
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Show original image
    st.image(img_array, caption="Original Image", use_column_width=True)

    # Pixelation parameters
    pixel_size = st.slider("Select pixel size", min_value=4, max_value=64, value=16, step=4)

    # Perform pixelation using OpenCV
    height, width = img_array.shape[:2]
    temp = cv2.resize(img_array, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_img = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    # Show pixelated image
    st.image(pixelated_img, caption="Pixelated Image", use_column_width=True)

    # Option to download
    result = Image.fromarray(pixelated_img)
    buf = BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Pixelated Image",
        data=byte_im,
        file_name="pixel_art.png",
        mime="image/png"
    )

else:
    st.info("⬆ Please upload an image to continue.")
