#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -U --user git+https://github.com/nilearn/nilearn.git
# os.listdir(os.getcwd())
#! pip show nilearn
#sys.path.append('/home/skim/.local/lib/python3.7/site-packages/nilearn')
#os.listdir('/home/skim/.local/lib/python3.7/site-packages/nilearn')
#!pip install -U --user nilearn
#!pip list -v
#!conda install -c conda-forge nilearn


# ### Import

# In[2]:


import os
import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import time
from glob import glob
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.stats import spearmanr

import nibabel as nib
from nibabel import load

from nilearn import plotting
from nilearn.datasets import load_mni152_template
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from nilearn.image import resample_to_img

from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball
from brainiak.fcma.preprocessing import prepare_searchlight_mvpa_data
from brainiak import io


# ### Preprocessing Files (BOLD, labels, etc)

# In[3]:


# # Default directory
# working_dir = '/bcbl/home/public/Consciousness/stats/Data'

# nums = [2]#list(set(range(1, 37)) - set([1, 18, 25, 26]))
# for num in nums:
#     subj = 'STATSMAR_{:02d}'.format(num)
#     # psychopy dir
#     psychopy_dir = sorted(glob(os.path.join(working_dir, subj, 'psychopy', 'Learning', '*.csv')))
#     # subj_dir
#     subj_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', subj)
#     labeled_csv = os.path.join(subj_dir, "preprocessed_events.csv")
#     csv_load = pd.read_csv(labeled_csv)
#     for n_run in range(8):
#         run_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', subj, str(n_run))
#         npy_file_dir = os.path.join(run_dir, 'preprocessed_whole_brain_BOLD.npy')
#         npy_load = np.load(npy_file_dir)
#         print(npy_load.shape)
        
#         cols = csv_load.columns.to_list()
#         for i in list(range(npy_load.shape[1])):
#             cols.append(i)
#         df = pd.DataFrame(np.column_stack([csv_load, npy_load]), columns = cols)
             
#         new_triple_num_li = []
#         triple_char_li = []
#         triple_group_li = []
        
#         for target in df['targets'].astype(int):
#             #print("target: ", target, "new_triple_num: ", triple_num_dic[target])
#             new_triple_num = triple_num_dic[target]
#             triple_char = triple_char_dic[target]
#             triple_group = triple_group_dic[target]
            
#             new_triple_num_li.append(new_triple_num)
#             triple_char_li.append(triple_char)
#             triple_group_li.append(triple_group)
        
#         df['new_triple_num'] = new_triple_num_li
#         df['triple_char'] = triple_char_li
#         df['triple_group'] = triple_group_li
#         df = df.sort_values(['triple_group', 'trials'], ascending=[True, True])
#         arg_sort_idx = df.index
        #print(processed_df.head())
        #print("argsort_idx: ", arg_sort_idx)
            
#         processed_BOLD = npy_load[np.argsort(arg_sort_idx)]
#         num_cols = df.columns[pd.to_numeric(df.columns, errors='coerce').to_series().notnull()]

#         df_mean = df.groupby(by='triple_char', as_index=False)[num_cols].apply(lambda x:x.mean())
#         print("Averaged_BOLD", df_mean.shape) # (24, 145537)
#         print("BOLD", processed_BOLD.shape) #(96, 145537)
#         print(subj, n_run)
        #np.save(os.path.join(run_dir, 'sorted_whole_brain.npy'), processed_BOLD)
        #np.save(os.path.join(run_dir, 'avg_events_brain.npy'), df_mean.to_numpy())


# ### Model

# In[8]:


default = np.ones((24, 24))

index = -1
for ind, val in enumerate(range(1, 9)):
    for order in range(3):
        index+=1
        default[index, index-order:index+3-order] = 0

mask = np.triu(np.ones_like(default, dtype=bool), k=0)

fig, ax = plt.subplots(figsize=(8, 8))
cmap = sns.mpl_palette("tab10", 2)
corr_plot = sns.heatmap(default, cmap=cmap, mask = mask, cbar=False, square=True, annot=True)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

legend_handles = [Patch(color=cmap[False], label='triple'),  # red
                   Patch(color=cmap[True], label='no_triple')]  # green
plt.legend(handles=legend_handles, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=15, handlelength=.8)

plt.setp(ax, ylim=(24, 0))
plt.setp(ax, xlim=(0, 24))

plt.show()


# ### NiftyMasker

# In[5]:


def normalize(data,axis = 1):
    return data - data.mean(axis).reshape(-1,1)
# Define voxel function
def sfn(l, msk, myrad, bcast_var):
    """
    l: BOLD
    msk: mask array
    myrad: not use
    bcast_var:  model
    """
    BOLD = l[0][msk,:].T.copy() # vectorize the voxel values in the sphere
#     print("BOLD", BOLD.shape) # <- for debugging
    model = bcast_var.copy() # vectorize the RDM
#     print("model", model.shape) # <- for debugging
    # pearson correlation
    RDM_X   = distance.pdist(normalize(BOLD),'correlation')
    RDM_y   = distance.pdist(normalize(model),'correlation')
    D,p     = spearmanr(RDM_X,RDM_y)
#     print("RDM_X", RDM_X.shape)
#     print("RDM_y", RDM_y.shape)
#     print("D", D, "p", p)
    return D


# In[10]:


#from nilearn.image import load_img
# # Default directory
working_dir = '/bcbl/home/public/Consciousness/stats/Data'

nums = [2]#list(set(range(1, 37)) - set([1, 18, 25, 26]))
for num in nums:
    subj = 'STATSMAR_{:02d}'.format(num)
    save_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'plot', subj)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    for n_run in range(1):
        run_folder = 'run'+str(n_run+1)
        run_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', subj, str(n_run))
        sorted_npy_file_dir = os.path.join(run_dir, 'sorted_whole_brain.npy')
        avg_npy_file_dir = os.path.join(run_dir, 'avg_events_brain.npy')
        npy_load = np.load(avg_npy_file_dir)
        #print(npy_load.shape) #(24, 145537)
        
        whole_mask_dir = os.path.join(working_dir, subj, run_folder, 'preprodef.feat', 'mask.nii.gz')

        if num < 28:
            example_func_dir = os.path.join(os.path.join(working_dir, subj, run_folder, 'preprodef.feat', 'example_func.nii.gz'))#filtered_func_data.nii.gz'))
        else:
            example_func_dir = os.path.join(working_dir, subj, run_folder+'.feat', 'ICA-AROMA.feat', 'example_func.nii.gz')#'filtered_func_data.nii.gz')
        #print(subj, '\n', example_func_dir)

#         whole_mask_dir = load_img(whole_mask_dir)
        masker = NiftiMasker(mask_img = None,).fit(example_func_dir)
        #example_func = load(example_func_dir)
        #print(load(whole_mask_dir).shape) # (88, 88, 64)
        #print(example_func.shape) #(88, 88, 64)
        
#         print("example_func_shape:", load(example_func_dir).shape)
#         print("whole_brain_dir_shape:", load(whole_mask_dir).shape)

        #npy_load = np.concatenate(npy_load)
        BOLD        = masker.inverse_transform(X = npy_load) # 153608
        #print("BOLD.shape", BOLD.shape)

        BOLD_image = np.asanyarray(BOLD.dataobj)
#         print(BOLD_image.shape)
        
        radius = 6 # in mm
        # Brainiak function
        sl = Searchlight(sl_rad = radius, 
                        max_blk_edge = radius - 1, 
                        shape = Ball,
                        min_active_voxels_proportion = 0,
                        )
        # distribute the data based on the sphere
        ## the first input is usually the BOLD signal, and it is in the form of 
        ## lists not arrays, representing each subject
        ## the second input is usually the mask, and it is in the form of array
        sl.distribute([BOLD_image],np.asanyarray(load(whole_mask_dir).dataobj) == 1)
        # broadcasted data is the data that remains the same during RSA
        sl.broadcast(default)
        # run searchlight algorithm
        global_outputs_no_tune = sl.run_searchlight(sfn,
                                                    pool_size = -1, # we run each RSA using a single CPU
                                                    )
        correlations_no_tune = new_img_like(example_func_dir,np.asanyarray(global_outputs_no_tune,dtype = np.float32))
        # masking
        correlations_no_tune = masker.inverse_transform(masker.transform_single_imgs(correlations_no_tune))
        plotting.plot_stat_map(correlations_no_tune,
                               #example_func_dir, #DimensionError: Input data has incompatible dimensionality: Expected dimension is 3D and you provided a 4D image. See http://nilearn.github.io/manipulating_images/input_output.html.
                       threshold = 1e-3,
                       draw_cross = False,
                       cmap = plt.cm.coolwarm,
                       vmax = .1,
                       title = 'Search Light for '+subj+' '+run_folder,
                       cut_coords = [0,0,25],
                       )

        plt.savefig(save_dir+"/"+'Search Light for '+subj+' '+run_folder+".png")


# In[ ]:




