#!/usr/bin/env python
# coding: utf-8

# This file accesses and downloads the NOAA GSOY yearly climate dataset and unites it into a single dataframe.

# In[1]:


import pandas as pd
import os
import wget


# In[2]:


wget.download('https://www.ncei.noaa.gov/data/global-summary-of-the-year/archive/gsoy-latest.tar.gz')


# In[ ]:


import tarfile

file = tarfile.open('gsoy-latest.tar.gz')
file.extractall('./gsoy-latest')
file.close()


# In[ ]:


root = os.getcwd()

root += '/gsoy-latest'
files = os.listdir(root)

all_dfs = []
for filename in files:
    if filename[0:2] == "US":
        path = root + "/" + filename
        new_df = pd.read_csv(path)
        print(filename + ' completed')
        all_dfs.append(new_df)

df = pd.concat(all_dfs)


# In[5]:


df = df.reset_index()


# Now, we restrict the data to the continental US.

# In[6]:


states_to_remove = ['VI', 'MP', 'AK','HI','PR','AS', 'GU']

df['STATE'] = df['NAME'].map(lambda x: x[-5:-3], na_action = 'ignore')
df = df[~df['STATE'].isin(states_to_remove)]


# We now remove a large number of unnecessary (and very infrequently reported) attributes.

# In[7]:


df = pd.concat([df[df.columns[:-139]],df[df.columns[-1:]]],axis=1)


# Some datapoints do not have listed entries for latitude and longitude. It is simplest to remove such data immediately. Likewise, we'll restrict the year to between 1950 and 2024, inclusive.

# In[8]:


df = df[~df.LATITUDE.isna()]
df = df[~df.LONGITUDE.isna()]
df = df.query('DATE >= 1950 and DATE <= 2024')


# In[9]:


df = df.reset_index()


# In[10]:


features_to_keep = df.columns[8:-1:2]


# In[ ]:


output_df = pd.concat([df[['DATE','LATITUDE','LONGITUDE','ELEVATION','NAME']],df[features_to_keep]],axis = 1)


# In[13]:


output_df.to_csv('yearly_climate_data.csv',index=False)

