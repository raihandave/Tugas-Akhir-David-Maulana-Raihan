#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

img = cv2.imread('Hilal.png')
original = img.copy()

xp = [0, 30, 60, 90, 128, 170, 192, 220, 255]
fp = [95, 40, 30, 40, 50, 120, 200, 150, 50]

x = np.arange(256)

table = np.interp(x, xp, fp).astype('uint8')
img = cv2.LUT(img, table)

cv2.imshow("original", original)
cv2.imshow("Output", img)

cv2.waitKey(0)
cv2.imwrite('cs_HILAL.png', img) 
cv2.imshow("Original image", img)
cv2.imshow('CS_Hilal.png', contrast_stretched)
cv2.destroyAllWindows() 


# In[ ]:


import cv2 
import numpy as np 

def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 
    
img = cv2.imread('Hilal.png') 
r1 = 10
s1 = 50
r2 = 10
s2 = 100

pixelVal_vec = np.vectorize(pixelVal)

contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2) 

cv2.imwrite('CS_Hilal.png', contrast_stretched) 
cv2.imshow("Original image", img)
cv2.imshow('CS_Hilal.png', contrast_stretched)
cv2.waitKey(0)
cv2.destroyAllWindows() 

