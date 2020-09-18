#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import warnings


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


columns=['user_id','movie_id','rating','timestamp']
df = pd.read_csv('movie_u.data',sep='\t',names=columns)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df['user_id'].nunique()


# In[8]:


df['movie_id'].nunique()


# In[9]:


movie_titles = pd.read_csv('movie_u.item',sep='\|', header=None)
movies_titles = movie_titles[[0,1]]


# In[10]:


movies_titles


# In[11]:


movies_titles.columns = ['movie_id','title']


# In[12]:


movies_titles.head()


# In[13]:


df = pd.merge(df,movies_titles, on="movie_id")


# In[14]:


df.head()


# # Exploratory Data Analysis

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df.groupby('title').mean()['rating'].sort_values(ascending=False)


# In[17]:


ratings = pd.DataFrame(df.groupby('title').mean()['rating'])


# In[18]:


ratings.head()


# In[19]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])


# In[20]:


ratings.sort_values(by='rating',ascending=False)


# In[21]:


plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'],bins=70,rwidth=0.5)
plt.show()


# In[22]:


plt.hist(ratings['rating'],bins=70,rwidth=0.85)
plt.show()


# In[23]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# In[26]:


moviemat = df.pivot_table(index="user_id",columns="title",values="rating")
# pivot_table = pandas matrix index = row, columns = column, values = data in that matrix


# In[27]:


moviemat.head()


# In[30]:


ratings.sort_values(by='num of ratings',ascending=False)


# In[31]:


starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()


# In[33]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[34]:


similar_to_starwars.head()


# In[39]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])


# In[40]:


corr_starwars.head()


# In[41]:


corr_starwars.dropna(inplace=True)


# In[42]:


corr_starwars.head()


# In[45]:


corr_starwars.sort_values('correlation',ascending=False).head(10)


# In[46]:


ratings.head()


# In[48]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[51]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)


# In[52]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie=corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions


# In[53]:


predictions = predict_movies("Titanic (1997)")


# In[54]:


predictions.head()


# In[ ]:




