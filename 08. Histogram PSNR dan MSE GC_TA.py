#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import locale
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter


# In[20]:


img3 = cv.imread('hilal.png', 1)
hist3 = cv.calcHist([img3], [0],  None, [256], [0,256])
plt.title('GC', size=15)
plt.xlabel('Nilai Piksel', size=15) 
plt.ylabel('Frekuensi', size=15)
plt.plot(hist3, color='green')
# Tambahkan minor ticks
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())


# In[150]:


import cv2
import numpy as np
from sewar import full_ref
from skimage import metrics


# In[151]:


ref_img = cv2.imread("hilal.png", 1)
img = cv2.imread("GC.png", 1)
img1 = cv2.imread("hilal.png", 1)


# In[152]:


import numpy
import math
import cv2

original = cv2.imread("hilal.png")
contrast = cv2.imread("GC.png")

def psnr(img1,img2):
    mse = numpy.mean((img1-img2)**2)
    if mse == 0:
        return 100
    print(mse)
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(original,contrast)
print(d)


# In[153]:


#MSE


# In[154]:


mse_skimg = metrics.mean_squared_error(ref_img, img)
print("MSE: based on scikit-image = ", mse_skimg)


# In[155]:


#RMSE


# In[156]:


rmse_skimg = metrics.normalized_root_mse(img1, img)
print("RMSE: based on scikit-image = ", rmse_skimg)


# In[157]:


# mose = rmse_skimg**2
# print(mose)


# In[158]:


# psnr_new = 10*math.log10(255 / mose)
# print(psnr_new)


# In[ ]:


# import cv2
# import numpy as np


# def gammaCorrection(src, gamma):
#     invGamma = 1 / gamma

#     table = [((i / 255) ** invGamma) * 255 for i in range(256)]
#     table = np.array(table, np.uint8)

#     return cv2.LUT(src, table)


# img = cv2.imread('ASLI1.png')
# gammaImg = gammaCorrection(img, 2.2)

# cv2.imshow('Original image', img)
# cv2.imshow('Gamma corrected image', gammaImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[ ]:


# import cv2 
# import numpy as np 
  
# # Function to map each intensity level to output intensity level. 
# def pixelVal(pix, r1, s1, r2, s2): 
#     if (0 <= pix and pix <= r1): 
#         return (s1 / r1)*pix 
#     elif (r1 < pix and pix <= r2): 
#         return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
#     else: 
#         return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 
  
# # Open the image. 
# img = cv2.imread('moonta.png') 
  
# # Define parameters. 
# r1 = 70
# s1 = 0
# r2 = 140
# s2 = 255
  
# # Vectorize the function to apply it to each value in the Numpy array. 
# pixelVal_vec = np.vectorize(pixelVal) 
  
# # Apply contrast stretching. 
# contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2) 
  
# # Save edited image. 
# cv2.imwrite('moontah.png', contrast_stretched) 


# In[ ]:


# import cv2 
# import matplotlib.pyplot as plt 
# import math
# import numpy as np 
# #this type of processing is suited for displaying image correctly for human eye based on monitor's display settings

# # Read an image 
# image = cv2.imread('ASLI1.png') 
# plt.imshow(image) 
# plt.show()


# In[ ]:


# # Histogram plotting of the image 
# color = ('b', 'g', 'r') 
  
# for i, col in enumerate(color): 
      
#     histr = cv2.calcHist([image],  
#                          [i], None, 
#                          [256],  
#                          [0, 256]) 
      
#     plt.plot(histr, color = col) 
      
#     # Limit X - axis to 256 
#     plt.xlim([0, 256]) 
      
# plt.show() 


# In[ ]:


# # Trying 4 gamma values. 
# for gamma in [0.1, 0.5, 1.2, 2.2, 3.2]: 

#     # Apply gamma correction. 
#     gamma_corrected = np.array(255*(image / 255) ** gamma, dtype = 'uint8') 
#     plt.imshow(gamma_corrected) 
#     plt.show() 


# In[ ]:


# # Histogram plotting of the 
# # log transformed image 
# color = ('b', 'g', 'r') 
  
# for i, col in enumerate(color): 
      
#     histr = cv2.calcHist([gamma_corrected],  
#                          [i], None, 
#                          [256], 
#                          [0, 256]) 
      
#     plt.plot(histr, color = col) 
#     plt.xlim([0, 256]) 
      
# plt.show()

