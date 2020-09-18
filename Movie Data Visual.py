#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv("movie_metadata.csv")


# In[6]:


df.head(n=10)


# In[7]:


df.columns


# In[8]:


titles = df.get('movie_title')
print(titles)


# In[9]:


print(type(titles))


# In[10]:


titles = list(df.get('movie_title'))


# In[11]:


print(type(titles))


# In[12]:


print(titles)


# In[15]:


print(titles[0])


# In[16]:


freq_titles={}

for title in titles:
    length=len(title)
    
    if freq_titles.get(length) is None:
        freq_titles[length] = 1
    else:
        freq_titles[length]+=1


# In[17]:


x = np.array(list(freq_titles.keys()))
y = np.array(list(freq_titles.values()))


# In[19]:


plt.style.use("seaborn")
plt.scatter(x,y)
plt.xlabel("Length of movie title")
plt.ylabel("Freq count")
plt.title("Movie data Visualization")
plt.show()


# In[ ]:




