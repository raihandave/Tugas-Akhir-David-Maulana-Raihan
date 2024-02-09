#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import locale
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter


# In[29]:


img2 = cv.imread("hilal.png", 1)
hist2 = cv.calcHist([img2], [1],  None, [256], [0,256])
plt.title('Citra Asli 1', size=15)
plt.xlabel('Nilai Piksel', size=15) 
plt.ylabel('Frekuensi', size=15)
plt.plot(hist2, color='black')
# Tambahkan minor ticks
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())


# In[32]:


img3 = cv.imread("Hilal.png", 1)
hist3 = cv.calcHist([img3], [1],  None, [256], [0,256])
plt.title('CLAHE 1', size=15)
plt.xlabel('Nilai Piksel', size=15) 
plt.ylabel('Frekuensi', size=15)
plt.plot(hist3, color='red')
# Tambahkan minor ticks
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())


# In[75]:


import cv2
import numpy as np
from sewar import full_ref
from skimage import metrics


# In[76]:


ref_img = cv2.imread("hilal.png", 1)
img = cv2.imread("Hilal.png", 1)
img1 = cv2.imread("hilal.png", 1)


# In[77]:


import numpy
import math
import cv2

original = cv2.imread("hilal.png")
contrast = cv2.imread("Hilal.png")

def psnr(img1,img2):
    mse = numpy.mean((img1-img2)**2)
    if mse == 0:
        return 100
    print(mse)
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(original,contrast)
print(d)


# In[78]:


#MSE


# In[79]:


mse_skimg = metrics.mean_squared_error(ref_img, img)
print("MSE: based on scikit-image = ", mse_skimg)


# In[80]:


#RMSE


# In[81]:


rmse_skimg = metrics.normalized_root_mse(img1, img)
print("RMSE: based on scikit-image = ", rmse_skimg)


# In[82]:


# mose = rmse_skimg**2
# print(mose)


# In[83]:


# psnr_new = 10*math.log10(255 / mose)
# print(psnr_new)


# In[41]:


#PSNR


# In[16]:


# psnr_skimg = metrics.peak_signal_noise_ratio(img1, img, data_range=None)
# print("PSNR: based on scikit-image = ", psnr_skimg)


# In[17]:


# psnr_skimg = metrics.peak_signal_noise_ratio(ref_img, img, data_range=None)
# print("PSNR: based on scikit-image = ", psnr_skimg)


# In[18]:


# import cv2
# img1 = cv2.imread('clahemoon.png')
# img2 = cv2.imread('moonta.png')
# psnr = cv2.PSNR(img1, img2)
# print(psnr)


# In[19]:


# import numpy as np
# import matplotlib.pyplot as plt
# import skimage
# from skimage.metrics import peak_signal_noise_ratio
# import cv2
 
# ref_img = cv2.imread('clahemoon.png')
# noisy_img = cv2.imread('moonta.png')
 
# PSNR = peak_signal_noise_ratio(ref_img,noisy_img)
# print(PSNR)


# In[20]:


# import math
# import cv2
# import numpy as np

# original = cv2.imread("moonta.png")
# contrast = cv2.imread("clahedoong.png", 1)

# def psnr(img1, img2):
#     mse = np.mean(np.square(np.subtract(img1.astype(np.int16),
#                                         img2.astype(np.int16))))
#     print(mse)
#     if mse == 0:
#         return np.Inf
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)  

# d = psnr(original, contrast)
# print(d)


# In[21]:


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


# In[22]:


# import numpy
# import cv2
# import xlwt

# #MSE--start
# def meanSquareError(img1, img2):
#     error = numpy.sum((img1.astype('float') - img2.astype('float')) ** 2)
#     error /= float(img1.shape[0] * img1.shape[1])
#     return error
# #MSE--end

# #PSNR--start
# def PSNR(img1, img2):
#     mse = meanSquareError(img1,img2)
#     if mse == 0:
#         return 100
#     return 10 * numpy.log10( numpy.square(255) / mse )
# #PSNR--end

# #comparison and write xls
# original = cv2.imread('moonta.png') #put your img
# lsbEncoded = cv2.imread('grayscale.png') #put your img
# dctEncoded = cv2.imread('clahemoon.png') #put your img

# book = xlwt.Workbook()
# sheet1=book.add_sheet("Sheet 1")
# style_string = "font: bold on , color red; borders: bottom dashed"
# style = xlwt.easyxf(style_string)
# sheet1.write(0, 0, "Original vs", style=style)
# sheet1.write(0, 1, "MSE", style=style)
# sheet1.write(0, 2, "PSNR", style=style)
# sheet1.write(1, 0, "LSB")
# sheet1.write(1, 1, meanSquareError(original, lsbEncoded))
# sheet1.write(1, 2, PSNR(original, lsbEncoded))
# sheet1.write(2, 0, "DCT")
# sheet1.write(2, 1, meanSquareError(original, dctEncoded))
# sheet1.write(2, 2, PSNR(original, dctEncoded))

# book.save("Comparis.xls")
# print("Comparison Results were saved as xls file!")
# #end


# In[ ]:




