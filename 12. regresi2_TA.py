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
y = df["C_Michelson"]


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


# In[11]:


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
ax.set_yticks([0.00, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125,  0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225,0.2375, 0.25])
# Add minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# reformat y-axis entries
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.scatter(x,y, s=12, c="blue")
plt.plot(x,yreg2, color="black") #kurva regresi
plt.title("Kontras Michelson dan Ketinggian Matahari", size=14)
plt.text(32, 0.015,
         r"$R^2 = 0,8767$",
         fontdict={"fontsize": "large", "horizontalalignment": "center"}) # by default posisi dinyatakan menggunakan koordinat data
plt.text(32, 0.03,
         r"$y =  0,7973e^{-0,0515x}$",
         fontdict={"fontsize": "large", "horizontalalignment": "center"}) # by default posisi dinyatakan menggunakan koordinat data
plt.xlabel('Ketinggian ☉ (°)', size=14)
plt.ylabel('Kontras Michelson (C)', size=14)
plt.gca().invert_xaxis()
plt.grid()


# In[ ]:




