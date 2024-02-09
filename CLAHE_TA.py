#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from skimage import io
from matplotlib import pyplot as plt


# In[2]:


img = cv2.imread("hilal.png", 1)
plt.imshow(img)


# In[3]:


lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab_img)


# In[4]:


plt.hist(l.flat, bins=100, range=(0,255))
equ = cv2.equalizeHist(l)


# In[5]:


plt.hist(equ.flat, bins=100, range=(0,255) )


# In[6]:


plt.imshow(equ, cmap='Greys')


# In[7]:


updated_lab_img1 = cv2.merge((equ,a,b))
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)


# In[8]:


clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
plt.hist(clahe_img.flat, bins=100, range=(0,255))


# In[11]:


update_lab_img2 = cv2.merge((clahe_img,a,b))


# In[12]:


CLAHE_img = cv2.cvtColor(update_lab_img2, cv2.COLOR_LAB2BGR)


# In[ ]:


cv2.imshow("Original image", img)
cv2.imshow("Equalized image", hist_eq_img)
cv2.imshow('CLAHE Image', CLAHE_img)
cv2.imwrite('grayscale.png', hist_eq_img)
cv2.imwrite('TA_CLAHE.png', CLAHE_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[18]:


# from math import log10, sqrt
# import cv2
# import numpy as np
  
# # original = cv2.imread("moonta.png")
# # compressed = cv2.imread("clahemoon.png", 1)

# def PSNR(original, compressed):
#     mse = np.mean((original - compressed) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse))
#     return psnr
  
# im1 = cv2.imread("moonta.png")
# im2 = cv2.imread("clahedoong.png", 1)
# # original = cv2.resize(im1,(300,300))
# # compressed = cv2.resize(im2,(300,300))
# value = PSNR(im1, im2)
# print(f"PSNR value is {value} dB")


# In[19]:


# # menyimpan gambar
# cv2.imwrite('claheversi2.png', CLAHE_img)


# In[20]:


# clahe 


# In[21]:


# import numpy as np
# import cv2 as cv


# In[22]:


# img = cv.imread('moonta.png',0)
# cv.imshow('Original Iamge', img)
# cv.waitKey(0)


# In[23]:


# clahe = cv.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv.imwrite('claheganteng.png',cl1)
# cv.imshow('CLAHE Enhaced Image', cl1)
# cv.waitKey(0)


# In[ ]:




