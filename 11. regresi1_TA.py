#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library dan package yang dibutuhkan
import pandas as pd #untuk dataframe
import matplotlib.pyplot as plt #untuk plotting
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression


# In[2]:


df =  pd.read_excel(r'hasil_kontras.xlsx')
df.head()


# In[3]:


x = df["tinggi"]
y = df["C_Ilyas"]


# In[4]:


x=x
ye=np.log(y)


# In[5]:


n = len(y) #jumlah data
xye=x*ye #xy
xx=x**2 #xx


# In[6]:


b=(n*xye.sum()-x.sum()*ye.sum())/(n*xx.sum()-(x.sum())**2)
b


# In[7]:


import math

#Menghitung a
A=ye.mean()-b*x.mean()
a=math.exp(A)
a


# In[8]:


print("Persamaan regresi pangkatnya adalah : y = {:.10f}e^{:.4f}x".format(a,b))


# In[9]:


yreg2=np.array(a*np.exp(b*x)) #yregresi


# In[10]:


dt=((y-y.mean())**2).sum()
d=((y-yreg2)**2).sum()
r2=np.sqrt((dt-d)/dt)
print("Nilai R adalah {:.4f} dan R^2 adalah {:.4f}".format(r2,r2**2))


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_xticks([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
ax.set_yticks([0.00, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6])
# Add minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# reformat y-axis entries
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.scatter(x,y, s=12, c="red")
plt.plot(x,yreg2, color="black") #kurva regresi
plt.title("Kontras Ilyas dan Ketinggian Matahari", size=14)
plt.text(32, 0.015,
         r"$R^2 = 0,8978$",
         fontdict={"fontsize": "large", "horizontalalignment": "center"}) # by default posisi dinyatakan menggunakan koordinat data
plt.text(32, 0.05,
         r"$y =  2,0710e^{-0,0551x}$",
         fontdict={"fontsize": "large", "horizontalalignment": "center"}) # by default posisi dinyatakan menggunakan koordinat data
plt.xlabel('Ketinggian ☉ (°)', size=14)
plt.ylabel('Kontras Ilyas (C)', size=14)
plt.gca().invert_xaxis()
plt.grid()

