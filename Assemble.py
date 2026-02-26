#!/usr/bin/env python
# coding: utf-8

# # Assemble results
# Ilya Balabin <ibalabin@avicenna-bio.com>

# In[1]:


import os, sys, glob
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

subdir = 'production_*'
f_fold = 'focus.csv.gz'
f_out = 'assembled.csv.gz'


# In[2]:


# assemble individual runs
df = pd.DataFrame()

for i, sd in enumerate(glob.glob(subdir)):
    try:
        df_one = pd.read_csv(os.path.join(sd, f_fold))
        df_one['Batch'] = i
        df = pd.concat([df, df_one], ignore_index=True)
    except:
        pass


# In[3]:


# get statistics
df_stat = df.groupby(['Fingerprint', 'V', 'Nsim']).agg({
    'AUC_bin': ['mean', 'std'], 
    'Enr_bin': ['mean', 'std']
}).reset_index()
df_stat.columns = df_stat.columns.map(lambda x: x[0] if x[1]=='' else '_'.join(x))
df_stat.to_csv(f_out, index=False)

