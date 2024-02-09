# -*- coding: utf-8 -*-
"""Untitled39.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFFU84aMhFCfh8K2S1kDbaEnwyf3ox6_
"""

# fast hue
import cv2
import numpy as np

def contrast_enhancement_fast_hue(image, factor=0.5):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase the value (V) channel
    hsv_image[:,:,2] = np.clip(factor * hsv_image[:,:,2], 0, 255)

    # Convert the image back to BGR color space
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image

# Read the input image
image = cv2.imread('hilaltadavid2.png')

# Apply contrast enhancement using the fast hue method
enhanced_image = contrast_enhancement_fast_hue(image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image (Fast Hue)', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()