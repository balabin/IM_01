#!/usr/bin/env python
# coding: utf-8

# # Deploy parallel focus calculations
# Ilya Balabin <ibalabin@avicenna-bio.com>
# 
# As is, this will generate 25 subfolders named "production_XXX" (X is a single digit).<br>
# In each subfolder, calculations with random split will repeat 20 times.<br>
# Altogether, the calculations will repeat 25 * 20 = 500 times.<br>
# To change any of these, edit the following parameters:<br>
# * n_repeats: the **total** number of calculations with random train-test split (default 500)<br>
# * n_batches: the number of parallel processes (**batches**) running concurrently (default 25)<br>
# 
# The number of runs per batch (default 20) will be calculated and saved automatically.<br>

# In[1]:


import os, sys, shutil, glob, stat

# total number of runs: n_clones * n_runs from Focus.py
n_repeats = 500
n_batches = 25
n_runs = int(n_repeats / n_batches)

# change path names to one level up
mapper = {
    "sys.path.insert(0, './')": "sys.path.insert(0, '../')",
    "incluster.csv": "../incluster.csv", 
    "outcluster.csv": "..//outcluster.csv", 
    "n_runs = 20": "n_runs = %d" % n_runs, 
}

f_run = 'Focus.py'
prod_template = 'production_%03d'


# In[2]:


with open (f_run, 'r') as f:
    source = f.read()

for key, value in mapper.items():
    source = source.replace(key, value)

for f in glob.glob("production_*"):
    shutil.rmtree(f, ignore_errors=True)

for i in range(0, n_batches):

    prod_dir = prod_template % i
    os.mkdir(prod_dir)
    f_out = os.path.join(prod_dir, f_run)
    with open(f_out, 'w') as f:
        f.write(source)
    os.chmod(f_out, 0o755)
    print(f_out, "done")

