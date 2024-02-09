#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[18]:


import numpy as np
import cv2 
from matplotlib import pyplot as plt
import locale
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter


# In[19]:


img3 = cv2.imread('CS.png', 1)
hist3 = cv2.calcHist([img3], [1],  None, [256], [0,256])
plt.title('CS', size=15)
plt.xlabel('Nilai Piksel', size=15) 
plt.ylabel('Frekuensi', size=15)
plt.plot(hist3, color='blue')
# Tambahkan minor ticks
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())


# In[4]:


import cv2
import numpy as np
from sewar import full_ref
from skimage import metrics


# In[5]:


ref_img = cv2.imread("hilal.png", 1)
img = cv2.imread("CS.png", 1)
img1 = cv2.imread("hilal.png", 1)


# In[6]:


import numpy
import math
import cv2

original = cv2.imread("hilal.png")
contrast = cv2.imread("CS.png")

def psnr(img1,img2):
    mse = numpy.mean((img1-img2)**2)
    if mse == 0:
        return 100
    print(mse)
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(original,contrast)
print(d)


# In[7]:


#MSE


# In[8]:


mse_skimg = metrics.mean_squared_error(ref_img, img)
print("MSE: based on scikit-image = ", mse_skimg)


# In[9]:


#RMSE


# In[10]:


rmse_skimg = metrics.normalized_root_mse(img1, img)
print("RMSE: based on scikit-image = ", rmse_skimg)


# In[11]:


# mose = rmse_skimg**2
# print(mose)


# In[12]:


# psnr_new = 10*math.log10(255 / mose)
# print(psnr_new)


# In[13]:


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# img =  cv2.imread('TA_ASLI_1.png')

# # The values of xp and fp can be varied to create custom tables as required 
# # and it will stretch the contrast even if min and max pixels are 0 and 255 

# xp = [0, 64, 128, 192, 255]
# fp = [0, 16, 128, 240, 255]

# x = np.arange(256)
# table = np.interp(x, xp, fp).astype('uint8')

# # cv2.LUT will replace the values of the original image with the values in the
# # table. For example, all the pixels having values 1 will be replaced by 0 and 
# # all pixels having values 4 will be replaced by 1.
# img = cv2.LUT(img, table)

# plt.imshow(img)
# plt.title('Contrast Stretched Image')

# # from shujaat
# def contrast_stretching(z, a, b, z1, zk):

#     new_array = np.copy(z)
        
#     for i,value in enumerate(z):
#         if value>=a and value<=b:
#             new_pixel_value = (((zk - z1)/(b-a))*value) + ((z1*b - zk*a)/(b-a))

#             new_array[i] = new_pixel_value

#     return new_array


# In[14]:


# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# from skimage import data, img_as_float
# from skimage import exposure


# matplotlib.rcParams['font.size'] = 8


# def plot_img_and_hist(image, axes, bins=256):
#     """Plot an image along with its histogram and cumulative histogram.

#     """
#     image = img_as_float(image)
#     ax_img, ax_hist = axes
#     ax_cdf = ax_hist.twinx()

#     # Display image
#     ax_img.imshow(image, cmap=plt.cm.gray)
#     ax_img.set_axis_off()

#     # Display histogram
#     ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
#     ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#     ax_hist.set_xlabel('Pixel intensity')
#     ax_hist.set_xlim(0, 1)
#     ax_hist.set_yticks([])

#     # Display cumulative distribution
#     img_cdf, bins = exposure.cumulative_distribution(image, bins)
#     ax_cdf.plot(bins, img_cdf, 'r')
#     ax_cdf.set_yticks([])

#     return ax_img, ax_hist, ax_cdf


# # Load an example image
# img = data.moon()

# # Contrast stretching
# p2, p98 = np.percentile(img, (2, 98))
# img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# # Equalization
# img_eq = exposure.equalize_hist(img)

# # Adaptive Equalization
# img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

# # Display results
# fig = plt.figure(figsize=(8, 5))
# axes = np.zeros((2, 4), dtype=object)
# axes[0, 0] = fig.add_subplot(2, 4, 1)
# for i in range(1, 4):
#     axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
# for i in range(0, 4):
#     axes[1, i] = fig.add_subplot(2, 4, 5+i)

# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
# ax_img.set_title('Low contrast image')

# y_min, y_max = ax_hist.get_ylim()
# ax_hist.set_ylabel('Number of pixels')
# ax_hist.set_yticks(np.linspace(0, y_max, 5))

# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
# ax_img.set_title('Contrast stretching')

# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
# ax_img.set_title('Histogram equalization')

# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
# ax_img.set_title('Adaptive equalization')

# ax_cdf.set_ylabel('Fraction of total intensity')
# ax_cdf.set_yticks(np.linspace(0, 1, 5))

# # prevent overlap of y-axis labels
# fig.tight_layout()
# plt.show()


# In[15]:


# import cv2
# import numpy as np

# # read image
# img = cv2.imread("moonta.png", cv2.IMREAD_COLOR)

# # normalize float versions
# norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# # scale to uint8
# norm_img1 = (255*norm_img1).astype(np.uint8)
# norm_img2 = np.clip(norm_img2, 0, 1)
# norm_img2 = (255*norm_img2).astype(np.uint8)

# # write normalized output images
# cv2.imwrite("zelda1_bm20_cm20_normalize1.png",norm_img1)
# cv2.imwrite("zelda1_bm20_cm20_normalize2.png",norm_img2)

# # display input and both output images
# cv2.imshow('original',img)
# cv2.imshow('normalized1',norm_img1)
# cv2.imshow('normalized2',norm_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[16]:


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# image = 'moonta.png'

# img = cv2.imread(image)

# a = 50.0
# ya = 30.0
# b = 150.0
# yb = 200.0
# l = 255.0

# alpha = int(input("Masukkan Nilai Alpha: "))
# beta = int(input("Masukkan Nilai Beta: "))
# gamma = int(input("Masukkan Nilai Gamma: "))

# img_gray2=np.where((img<a), np.floor(alpha*img),
#     np.where(((img>a)&(img<b)), np.floor(beta*(img-a)+ya),
#         np.where(((img>b)&(img<l)), np.floor(gamma*(img-b)+yb),img)))

# print(img_gray2)
# print(img_gray2.shape)

# cv2.imwrite(image+'grayafff.png',img_gray2)

# plt.show()
# plt.hist(img_gray2.ravel(),256,[0,256]); plt.show()


# In[17]:


# import cv2 
# import matplotlib.pyplot as plt 
# import math
# import numpy as np 
# '''
# Contrast stretching (often called normalization) is a 
# simple image enhancement technique that attempts to 
# improve the contrast in an image by `stretching' 
# the range of intensity values it contains 
# to span a desired range of values,
# e.g. the the full range of pixel values that the
# image type concerned allows. 
# It differs from the more sophisticated histogram equalization 
# in that it can only apply a linear scaling function to the 
# image pixel values. As a result the `enhancement' is less harsh. 
# (Most implementations accept a graylevel image as input and produce another graylevel image as output.)
# '''

# # Read an image 
# image = cv2.imread('ASLI1.png') 
# plt.imshow(image) 
# plt.show()


# In[18]:


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


# In[19]:


# # Apply contrast stretching method 
# maxiI = 200
# miniI = 3

# maxoI = 100 
# minoI = 3

# stretched_image = image.copy()
# # get height and width of the image 
# height, width, _ = image.shape 
  
# for i in range(0, height - 1): 
#     for j in range(0, width - 1): 
          
#         # Get the pixel value 
#         pixel = stretched_image[i, j] 
          
#         # scale each pixel by this formula
#         '''
#         pout = (pin - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI 
        
#         '''
        
          
#         # 1st index contains red pixel 
#         pixel[0] = (pixel[0] - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI  
          
#         # 2nd index contains green pixel 
#         pixel[1] = (pixel[1] - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI 
          
#         # 3rd index contains blue pixel 
#         pixel[2] = (pixel[2] - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI 
          
#         # Store new values in the pixel 
#         stretched_image[i, j] = pixel 
  

# #original image
# plt.imshow(image) 
# plt.show()

# #stretched image
# plt.imshow(stretched_image) 
# plt.show() 


# In[20]:


# # Histogram plotting of the image 
# color = ('b', 'g', 'r') 
  
# for i, col in enumerate(color): 
      
#     histr = cv2.calcHist([stretched_image],  
#                          [i], None, 
#                          [256],  
#                          [0, 256]) 
      
#     plt.plot(histr, color = col) 
      
#     # Limit X - axis to 256 
#     plt.xlim([0, 256]) 
      
# plt.show() 


# In[21]:


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt


# In[22]:


# img_gray = cv2.imread('moonta.png',0)


# In[23]:


# #parameter
# a = 50  #nilai min
# b = 200 #nilai max
# c = np.min(img_gray)
# d = np.max(img_gray)


# In[24]:


# #rumus
# img_cont = ((img_gray-c) * ((b-a) / (d-c))) + a
# img_cont = np.uint8(img_cont)


# In[25]:


# #histogram
# hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# hist_cont = cv2.calcHist([img_cont], [0], None, [256], [0, 256])


# In[26]:


# plt.figure(dpi=120,figsize=(10,6))


# In[27]:


# plt.subplot(221)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Citra awal")


# In[28]:


# plt.subplot(223)
# plt.bar(range(256), hist_gray.flatten(), label='Citra asli',
#        color="red")
# plt.xlabel("Intensitas piksel")
# plt.ylabel("Jumlah piksel")
# plt.title("Citra awal")



# In[29]:


# plt.subplot(222)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Citra stretching")


# In[30]:


# plt.subplot(224)
# plt.bar(range(256), hist_cont.flatten(), label='Citra stretching',
#        color="red")
# plt.xlabel("Intensitas piksel")
# plt.ylabel("Jumlah piksel")
# plt.title("Citra stretching")



# In[31]:


# plt.tight_layout()
# plt.show()


# In[32]:


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# img =  cv2.imread('ASLI1.png')

# # The values of xp and fp can be varied to create custom tables as required 
# # and it will stretch the contrast even if min and max pixels are 0 and 255 

# xp = [0, 14, 128, 192, 255]
# fp = [0, 16, 100, 200, 225]

# x = np.arange(256)
# table = np.interp(x, xp, fp).astype('uint8')

# # cv2.LUT will replace the values of the original image with the values in the
# # table. For example, all the pixels having values 1 will be replaced by 0 and 
# # all pixels having values 4 will be replaced by 1.
# # img = cv2.LUT(img, table)


# plt.imshow(img)
# cv2.imwrite('csta1.png',0)
# # plt.title('Contrast Stretched Image')

# # from shujaat
# # def contrast_stretching(z, a, b, z1, zk):

# #     new_array = np.copy(z)
        
# #     for i,value in enumerate(z):
# #         if value>=a and value<=b:
# #             new_pixel_value = (((zk - z1)/(b-a))*value) + ((z1*b - zk*a)/(b-a))

# #             new_array[i] = new_pixel_value

# #     return new_array
     


# In[ ]:




