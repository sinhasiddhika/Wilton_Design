from os import path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pyxelate import Pyx
from PIL import Image
import cv2

SAVE_IMAGES = True  # To allow saving

def plot(subplots=[], save_as=None, fig_h=9):
    """Plotting helper function"""
    fig, ax = plt.subplots(int(np.ceil(len(subplots) / 3)),
                           min(3, len(subplots)),
                           figsize=(18, fig_h))
    if len(subplots) == 1:
        ax = [ax]
    else:
        ax = ax.ravel()
    for i, subplot in enumerate(subplots):
        if isinstance(subplot, dict):
            ax[i].set_title(subplot["title"])
            ax[i].imshow(subplot["image"])
        else:
            ax[i].imshow(subplot)
    fig.tight_layout()
    if save_as is not None and SAVE_IMAGES:
        plt.savefig(path.join("./", save_as), transparent=True)
    plt.show()

# Load the image
image = io.imread("14.jpg")  # Make sure this path is correct

# Desired output dimensions
desired_width = 128
desired_height = 128

# Downsample factor
h, w = image.shape[:2]
downsample_by = max(w // desired_width, h // desired_height)

# Pixelation
palette = 7
pyx = Pyx(factor=downsample_by, palette=palette)
pyx.fit(image)
pixel_art = pyx.transform(image)

# Resize to target
pixel_art_resized = cv2.resize(pixel_art, (desired_width, desired_height), interpolation=cv2.INTER_NEAREST)

# Save result
output_path = "pixel_art_output.png"
io.imsave(output_path, pixel_art_resized)

# Show original vs pixelated
plot([image, pixel_art_resized], save_as="comparison_plot.png")
print(f"Pixel art saved to: {output_path}")
