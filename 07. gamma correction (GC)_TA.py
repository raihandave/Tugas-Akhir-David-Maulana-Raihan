#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 
import numpy as np 
  
# Open the image. 
img = cv2.imread('Hilal.png') 
  
# Trying 4 gamma values. 
for gamma in [1.0]: 
      
    # Apply gamma correction. 
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8') 
  
    # Save edited images. 
    cv2.imwrite('GC_TA.png', gamma_corrected) 


# In[ ]:


cv2.imshow("Original image", img)
cv2.imshow('GC_TA.png', gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows() 

