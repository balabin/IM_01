#!/usr/bin/env python
# coding: utf-8

# # Focus with Akt cluster 6
# Ilya Balabin <ibalabin@avicenna-bio.com>

# In[ ]:


import os, sys, copy
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

sys.path.insert(0, '../')
from Fingerprints import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import sklearnex

# fingerprints
fingerprints = [ 'morgan3' ]
n_bits = 1024

# decision values
V = np.arange(2.5, 51., 2.5)

# Nsim values
Nsim = np.arange(0, 2226, 25)

# do not change this; edit Deploy.py instead
n_runs = 20

f_cluster = 'incluster.csv'
f_notcluster = 'outcluster.csv'
f_fold = 'focus.csv.gz'


# In[ ]:


# calculate enrichment
def enrichment_score(y_test, confs):

    # sort test chemicals by prediction confidence
    y_pred = (confs>0.5).astype(int)
    idx = np.argsort(confs)[::-1]
    y_test, y_pred = y_test[idx], y_pred[idx]

    # enrichment calculated on top 10% of predicted activities
    n10 = int(0.1 * y_test.shape[0])
    return np.sum(y_test[:n10]) / np.sum(y_test)


# In[ ]:


# read data
df_in = pd.read_csv(f_cluster)
df_in['tag'] = 'in'

df_out = pd.read_csv(f_notcluster)
df_out['tag'] = 'out'

df = pd.concat([df_in, df_out])
df = df[['tag', 'smiles', 'IC50']]
del(df_in, df_out)


# In[ ]:


# focus calculations
data_fold = []
for fp in fingerprints:

    df['bv_fp'] = FingerprintsFromSmiles(df.smiles, fp, size=n_bits)
    df_in, df_out = df.loc[df.tag=='in'].copy(), df.loc[df.tag=='out'].copy()

    df['sims'] = [[DataStructs.TanimotoSimilarity(fp1, fp2) for fp2 in df_in.bv_fp] 
                  for fp1 in tqdm(df.bv_fp, desc='Tanimoto sims', leave=False)]

    # calculation runs
    df_in, df_out = df.loc[df.tag=='in'].copy(), df.loc[df.tag=='out'].copy()
    for run in tqdm(range(0, n_runs), desc='Run', leave=False):

        # random split
        df_train_in, df_test = train_test_split(df_in, test_size=0.5)

        # sort by reverse average similarity to the training set 
        sorting_sims = np.average(np.stack(df_out.sims)[:, df_train_in.index], axis=1)
        idx = sorting_sims.argsort()[::-1]
        df_out = df_out.reset_index(drop=True, inplace=False)
        df_out = df_out.reindex(idx)
        df_train = pd.concat([df_train_in, df_out], ignore_index=True)

        X_train_full, X_test = np.stack(df_train.bv_fp), np.stack(df_test.bv_fp)

        # decision value
        for cutoff in tqdm(V, desc='Decision value', leave=False):

            y_train_full, y_test = (df_train.IC50<=cutoff).astype(int).to_numpy(), (df_test.IC50<=cutoff).astype(int).to_numpy()

            # training set outmentation size
            for ns in tqdm(Nsim, desc='Nsim', leave=False):

                X_train, y_train = X_train_full[:len(df_train_in)+ns, :], y_train_full[:len(df_train_in)+ns]

                # exceptions possible when the training set only includes one class
                try:

                    # training and inference
                    model = RandomForestClassifier(class_weight=None, 
                                                   n_estimators=1000, 
                                                   max_depth=30)
                    y_proba = model.fit(X_train, y_train).predict_proba(X_test)
                    conf = y_proba.T[1]
                    del(model)

                    # accuracy metrics
                    conf = (conf>0.5).astype(int)
                    auc_bin = roc_auc_score(y_test, conf)
                    enr_bin = enrichment_score(y_test, conf)
                    data_fold.append([fp, run, cutoff, ns, auc_bin, enr_bin])

                except:
                    data_fold.append([fp, run, cutoff, ns, None, None])

            # create checkpoint
            df_fold = pd.DataFrame(data=data_fold, columns=['Fingerprint', 'Run', 'V', 'Nsim', 'AUC_bin', 'Enr_bin'])
            df_fold.to_csv(f_fold, index=False)

