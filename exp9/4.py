# Write a Python program to simulate sharpening using spatial high pass filters.

import cv2
import numpy as np

# Load the image
image = cv2.imread('309582-1920x1080-desktop-full-hd-spider-man-background-photo.jpg')

# Apply Gaussian smoothing to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Create a Laplacian kernel for sharpening
laplacian_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Apply the Laplacian filter for sharpening
sharpened_image = cv2.filter2D(blurred_image, -1, laplacian_kernel)

# Display the images
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Sharpened Image", sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()