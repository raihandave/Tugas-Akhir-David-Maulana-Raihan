# -*- coding: utf-8 -*-
"""Untitled39.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFFU84aMhFCfh8K2S1kDbaEnwyf3ox6_
"""

import cv2

# Read the input image
image_path = 'Hilal2.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create an MSER object with custom parameters
mser = cv2.MSER_create(delta=5, min_area=100, max_area=2000, max_variation=0.25)

# Detect regions in the image
regions, _ = mser.detectRegions(gray_image)

# Draw the detected regions on the original image
image_with_regions = image.copy()
for region in regions:
    color = (0, 0, 255)  # Red color
    cv2.polylines(image_with_regions, [region], isClosed=True, color=color, thickness=1)

# Display the original image and the image with detected regions
cv2.imshow('Original Image', image)
cv2.imshow('Image with MSER Regions', image_with_regions)
cv2.imwrite('mser_detection_2.png', image_with_regions)
cv2.waitKey(0)
cv2.destroyAllWindows()