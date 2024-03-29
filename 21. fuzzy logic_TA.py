# -*- coding: utf-8 -*-
"""Untitled39.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFFU84aMhFCfh8K2S1kDbaEnwyf3ox6_
"""

# #fuzzy logic
import cv2
import numpy as np
import skfuzzy as fuzz

def fuzzy_contrast_enhancement(image):
    # Convert the image to a single-channel grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define fuzzy membership functions
    x = np.arange(0, 256, 1)

    # Input fuzzy sets
    dark = fuzz.trimf(x, [0, 0, 100])
    mid = fuzz.trimf(x, [60, 120, 161])
    bright = fuzz.trimf(x, [127, 205, 255])

    # Apply fuzzy rules
    dark_membership = fuzz.interp_membership(x, dark, gray_image)
    mid_membership = fuzz.interp_membership(x, mid, gray_image)
    bright_membership = fuzz.interp_membership(x, bright, gray_image)

    # Define output fuzzy set
    enhanced_contrast = np.zeros_like(gray_image)

    # Apply fuzzy enhancement rules
    enhanced_contrast[(gray_image >= 0) & (gray_image <= 127)] = 0.5 * gray_image[(gray_image >= 0) & (gray_image <= 127)]
    enhanced_contrast[(gray_image > 127) & (gray_image <= 191)] = 2 * gray_image[(gray_image > 127) & (gray_image <= 191)]
    enhanced_contrast[(gray_image > 191) & (gray_image <= 255)] = 1.5 * gray_image[(gray_image > 191) & (gray_image <= 255)]

    return enhanced_contrast.astype(np.uint8)

# Read the input image
image = cv2.imread('Hilal.png')

# Apply fuzzy logic contrast enhancement
enhanced_image = fuzzy_contrast_enhancement(image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Fuzzy Enhanced Image', enhanced_image)
cv2.imwrite('fuzzy_hilal.png', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()