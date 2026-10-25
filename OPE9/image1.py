import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load the Image
# -------------------------
src_image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

if src_image is None:
    print("Image not found!")
    exit()

# -------------------------
# Gaussian Smoothing
# -------------------------
smoothed_image = cv2.GaussianBlur(
    src_image,
    (5, 5),
    sigmaX=1
)

# -------------------------
# Sharpening
# -------------------------
sharpening_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

sharpened_image = cv2.filter2D(
    smoothed_image,
    -1,
    sharpening_kernel
)

# -------------------------
# Display Results
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(src_image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(smoothed_image, cmap='gray')
plt.title("Gaussian Smoothed")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened_image, cmap='gray')
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()