#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

def apply_clahe(image, clip_limit=12.0, tile_grid_size=(8, 8)):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)
    
    clahe_image = clahe.apply(gray)
    
    return clahe_image

image = cv2.imread('Hilal.png')

clahe_image = apply_clahe(image)

cv2.imshow('Original Image', image)
cv2.imshow('CLAHE Image', clahe_image)
cv2.imwrite('HILAL_CLAHE.png', clahe_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




