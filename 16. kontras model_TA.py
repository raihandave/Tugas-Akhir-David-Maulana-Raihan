#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #untuk dataframe
import matplotlib.pyplot as plt #untuk plotting
import numpy as np


# In[2]:


df =  pd.read_excel(r'hasil_kontras_model.xlsx')
df.head()


# In[3]:


cilyas = df["C_Ilyas"]
cmic = df["C_Michelson"]
hsun = df["tinggi"]
hmoon = df["h_m"]
el = df["Elongasi"]
age = df["Umur"]
ilum = df['Iluminasi']
lebar = df['Moon_Width']
ARCV = df["ARCV"]
DAZ = df["DAZ"]


# In[23]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = hsun
x = hmoon

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)

# Creating color map
my_cmap = plt.get_cmap('cool')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Ilyas, Ketinggian ☉, Ketinggian ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel('h☾ (°)', size=12)
ax.set_zlabel('Kontras Ilyas (C)', size=12, rotation=180)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_1.png",dpi=600)


# In[42]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = hsun
x = hmoon

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('cool')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Michelson, Ketinggian ☉, Ketinggian ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel('h☾ (°)', size=12)
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_2.png",dpi=600)


# In[25]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = hsun
x = el

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('viridis')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("3D Plot kontras Ilyas, Ketinggian ☉, Elongasi",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel('Elongasi (°)', size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_3.png",dpi=600)



# In[43]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = hsun
x = el

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('viridis')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("3D Plot kontras Michelson, Ketinggian ☉, Elongasi",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel('Elongasi (°)', size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_4.png",dpi=600)



# In[28]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = hsun
x = age

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('Wistia')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Ilyas, Ketinggian ☉, Umur ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel(' Umur ☾ (Hari)', size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_5.png",dpi=600)



# In[44]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = hsun
x = age

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('Wistia')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Michelson, Ketinggian ☉, Umur ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel(' Umur ☾ (Hari)', size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_6.png",dpi=600)



# In[30]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = hsun
x = ilum

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('PuOr')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Ilyas, Ketinggian ☉, Iluminasi ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel('Iluminasi ☾ (%)', size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_7.png",dpi=600)





# In[45]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = hsun
x = ilum

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('PuOr')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Michelson, Ketinggian ☉, Iluminasi ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel('Iluminasi ☾ (%)', size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_8.png",dpi=600)





# In[34]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = hsun
x = lebar

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("3D Plot kontras Ilyas, Ketinggian ☉, Lebar ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel("Lebar ☾ (')", size=12, labelpad=7)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.02, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_9.png",dpi=600)


# In[46]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = hsun
x = lebar

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

plt.title("3D Plot kontras Michelson, Ketinggian ☉, Lebar ☾",fontweight='bold', size=16)
ax.set_ylabel('h☉ (°)', size=12)
ax.set_xlabel("Lebar ☾ (')", size=12, labelpad=7)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.02, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_10.png",dpi=600)


# In[39]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = el
x = ARCV

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('brg')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("           3D Plot kontras Ilyas, Elongasi, ARCV",fontweight='bold', size=16)
ax.set_ylabel('Elongasi (°)', size=12)
plt.gca().invert_yaxis()
ax.set_xlabel("ARCV (°)", size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_11.png",dpi=600)



# In[41]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = el
x = ARCV

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('brg')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))

plt.title("3D Plot kontras Michelson, Elongasi, ARCV",fontweight='bold', size=16)
ax.set_ylabel('Elongasi (°)', size=12)
plt.gca().invert_yaxis()
ax.set_xlabel("ARCV (°)", size=12)
plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_12.png",dpi=600)



# In[48]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')


# Creating dataset
z = cilyas
y = el
x = DAZ

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('cividis')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("           3D Plot kontras Ilyas, Elongasi, DAZ",fontweight='bold', size=16)
ax.set_ylabel('Elongasi (°)', size=12)
plt.gca().invert_yaxis()
ax.set_xlabel("DAZ (°)", size=12)
# plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_13.png",dpi=600)




# In[51]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = el
x = DAZ

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('cividis')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("        3D Plot kontras Michelson, Elongasi, DAZ",fontweight='bold', size=16)
ax.set_ylabel('Elongasi (°)', size=12)
plt.gca().invert_yaxis()
ax.set_xlabel("DAZ (°)", size=12)
# plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
cb.set_label('Kontras', size=16)
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_14.png",dpi=600)




# In[53]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cilyas
y = ARCV
x = DAZ

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('summer')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("                3D Plot kontras Ilyas, ARCV, DAZ",fontweight='bold', size=16)
ax.set_ylabel('ARCV (°)', size=12)
plt.gca().invert_yaxis()
ax.set_xlabel("DAZ (°)", size=12)
# plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Ilyas (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_15.png",dpi=600)




# In[54]:


# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale

# apply German locale settings
locale.setlocale(locale.LC_ALL, 'de_DE')

# Creating dataset
z = cmic
y = ARCV
x = DAZ

# Creating figure
fig = plt.figure(figsize = (12, 6))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.3,
		alpha = 0.2)


# Creating color map
my_cmap = plt.get_cmap('summer')

# Creating plot
sctt = ax.scatter3D(x, y, z,
					alpha = 0.8,
					c = (z),
					cmap = my_cmap,
					marker ='.')
ax.view_init(elev=30, azim=40)

ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.3n}'))

plt.title("           3D Plot kontras Michelson, ARCV, DAZ",fontweight='bold', size=16)
ax.set_ylabel('ARCV (°)', size=12)
plt.gca().invert_yaxis()
ax.set_xlabel("DAZ (°)", size=12)
# plt.gca().invert_xaxis()
ax.set_zlabel('Kontras Michelson (C)', size=12)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspe,ct = 10, label="Kontras")
cb = fig.colorbar(sctt, orientation='vertical',aspect = 30, pad= 0.01, label='Kontras')
cb.set_label('Kontras', size=16)
cb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.1n}'))
# fig.set_label(size='large', weight='bold')
# show plot
plt.show()
fig.savefig("3dhilal_16.png",dpi=600)




# In[ ]:




