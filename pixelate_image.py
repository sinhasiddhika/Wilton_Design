import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pyxelate import Pyx
from PIL import Image
import cv2
import os
from io import BytesIO

# Streamlit page settings
st.set_page_config(page_title="Pixelate Image with Pyxelate", layout="centered")

st.title("🎨 Pixelate Your Image with Pyxelate")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show original image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    # Set parameters
    desired_width = st.slider("Desired Width", min_value=16, max_value=512, value=128)
    desired_height = st.slider("Desired Height", min_value=16, max_value=512, value=128)
    palette = st.slider("Number of Colors (Palette)", min_value=4, max_value=16, value=7)

    # Process button
    if st.button("Pixelate Image"):
        # Downsampling factor
        h, w = image_np.shape[:2]
        downsample_by = max(w // desired_width, h // desired_height)

        # Pixelate
        pyx = Pyx(factor=downsample_by, palette=palette)
        pyx.fit(image_np)
        pixel_art = pyx.transform(image_np)

        # Resize to target
        pixel_art_resized = cv2.resize(pixel_art, (desired_width, desired_height), interpolation=cv2.INTER_NEAREST)

        # Save output
        output_path = "pixel_art_output.png"
        Image.fromarray(pixel_art_resized).save(output_path)

        st.image(pixel_art_resized, caption="🧩 Pixelated Image", use_column_width=True)

        # Download button
        buf = BytesIO()
        Image.fromarray(pixel_art_resized).save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("Download Pixelated Image", data=byte_im, file_name="pixel_art_output.png", mime="image/png")
