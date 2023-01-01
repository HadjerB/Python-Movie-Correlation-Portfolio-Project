#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import Libraries

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8) # Adjusts the configuration of the plots we will create 

# Read in the data 

import os 
os.getcwd
os.chdir('/Users/HadjerBoukhatem/Downloads')

df= pd.read_csv(r'/Users/HadjerBoukhatem/Downloads/movies 2.csv')


# In[28]:


# Let's look at the data

df.head(50)


# In[59]:


# Let's see if there is any missing data 

for col in df.columns:
    pct_missing= np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))
    
    
df = df.dropna()
    


# In[34]:


# Data Types for our columns 

df.dtypes


# In[37]:


# Change data type of columns 

df['budget'] = df['budget'].astype('int64')

df['gross'] = df['gross'].astype('int64')


# In[38]:





# In[42]:


# Create correct year column 

df['yearcorrect']=df['released'].str.extract(pat='([0-9]{4})').astype(int)


# In[48]:


df


# In[44]:


df.sort_values(by=['gross'],inplace= False, ascending=False)


# In[47]:


pd.set_option('display.max_rows', None)


# In[77]:


# Drop any duplicates 

df.drop_duplicates()

df.head()


# In[ ]:


# Let's start looking at correlation


# In[70]:


df.corr(method='pearson')

df.corr(method='kendall')

df.corr(method='spearman')


# In[ ]:


# High correlation between budget and gross


# In[66]:


correlation_matrix= df.corr(method='pearson')


# In[68]:


sns.heatmap(correlation_matrix,annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show


# In[69]:


correlation_mat = df.corr(method='pearson')



corr_pairs = correlation_mat.unstack()

corr_pairs


# In[73]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[75]:


high_corr= sorted_pairs[(sorted_pairs)>0.5]

high_corr


# In[ ]:


# Votes and budget have the highest coorelation to gross earnings

