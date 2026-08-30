# Write a Python program to simulate smoothening and sharpening operation on image using spatial filter.

import cv2
import numpy as np

# Load the image
image = cv2.imread('309582-1920x1080-desktop-full-hd-spider-man-background-photo.jpg')

# Define Gaussian kernel for smoothing
kernel_size = (5, 5)
sigma = 1.5

gaussian_kernel = cv2.getGaussianKernel(5, sigma)
gaussian_kernel = np.outer(gaussian_kernel, gaussian_kernel)

# Apply Gaussian smoothing
smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)

# Define sharpening kernel
sharpening_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

# Apply sharpening
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Smoothed Image", smoothed_image)
cv2.imshow("Sharpened Image", sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()