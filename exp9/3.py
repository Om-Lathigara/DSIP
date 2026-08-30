# Write a Python program to simulate smoothening using median filters.

import cv2
import numpy as np

# Load the image
image = cv2.imread('309582-1920x1080-desktop-full-hd-spider-man-background-photo.jpg')

# Define the size of the median filter kernel (should be an odd number)
kernel_size = 5

# Apply the Median filter for smoothing
smoothed_image = cv2.medianBlur(image, kernel_size)

# Display the original and smoothed images
cv2.imshow("Original Image", image)
cv2.imshow("Smoothed Image", smoothed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()