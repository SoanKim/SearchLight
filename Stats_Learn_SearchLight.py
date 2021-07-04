#!/usr/bin/env python
# coding: utf-8
"""
Created on Fri Jun 25 13:53:56 2021

@author: nmei, skim

"""
################################### Import ####################################

import os
import sys 
import time
from   glob               import glob
from   shutil             import copyfile

import numpy              as np
import pandas             as pd
import string
import seaborn            as sns
import matplotlib.pyplot  as plt
from   matplotlib.patches import Patch

from   scipy.spatial      import distance
from   scipy.stats        import spearmanr

import nibabel            as nib
from   nibabel            import load

from nilearn              import plotting
from nilearn.input_data   import NiftiMasker
from nilearn.image        import new_img_like
from nilearn.image        import resample_to_img
from nilearn.image        import load_img
from nilearn.image        import index_img
from nilearn.image        import concat_imgs
from nilearn.signal       import clean    as clean_signal

from brainiak.searchlight.searchlight     import Searchlight
from brainiak.searchlight.searchlight     import Ball
from brainiak.fcma.preprocessing          import prepare_searchlight_mvpa_data
from brainiak                             import io

######################### DataFrame Helper Functions ##########################

def add_track(df_sub):
    n_rows = df_sub.shape[0]
    if len(df_sub.index.values) > 1:
        temp = '+'.join(str(item + 10) for item in df_sub.index.values)
    else:
        temp = str(df_sub.index.values[0])
    df_sub = df_sub.iloc[0,:].to_frame().T # why did I use 1 instead of 0?
    df_sub['n_volume'] = n_rows
    df_sub['time_indices'] = temp
    return df_sub

def groupby_average(fmri,df,groupby = ['trials.thisN']):
    BOLD_average = np.array([np.mean(fmri[df_sub.index],0) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([add_track(df_sub) for ii,df_sub in df.groupby(groupby)])
    return BOLD_average,df_average

### STATSLEARN Helper Functions (Without this, it's hard to group the symbols) #####

triple = [5, 19, 3, 17, 23, 4, 24, 6, 21, 13, 16, 15, 11, 22, 8, 2, 18, 20, 14, 1, 10, 12, 7, 9]
triple_char = list(string.ascii_lowercase)[:24]
triple_char_dic = dict(zip(triple, triple_char))

triple_num = list(range(24))
triple_num_dic = dict(zip(triple, triple_num))

triple_group = []
for i in list(range(8)):
    triple_group.extend([i]*3)
triple_group_dic = dict(zip(triple, triple_group))

################################## Model I ####################################

model = np.eye(24)

index = -1
for ind, val in enumerate(range(1, 9)):
    for order in range(3*1):
        index+=1
        model[index, index-order:index+3*1-order] = 1

mask = np.triu(np.ones_like(model, dtype=bool), k=0)

fig, ax = plt.subplots(figsize=(8, 8))
cmap = sns.mpl_palette("tab10", 2)
RSA_plot = sns.heatmap(model, cmap=cmap, mask = mask, cbar=False, square=True, annot=True, linewidths=.1)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Legend
legend_handles = [Patch(color=cmap[False], label='triple'),
                   Patch(color=cmap[True], label='no_triple')]
plt.legend(handles=legend_handles, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=15, handlelength=.8)
plt.setp(ax, ylim=(24, 0))
plt.setp(ax, xlim=(0, 24))
#plt.show()

######################## Searchlight Helper Function ##########################

def RDM_model_func():
    columns_li = []
    for column_length in range(21, -3, -3):
	    for ones in range(2, -1, -1):
	        each_column = np.concatenate([np.ones(ones), np.zeros(column_length)])
	        columns_li.append(each_column)
        
    RDM_model = np.concatenate(columns_li[:-1])
    return RDM_model

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
    model = bcast_var.copy() # vectorize the RDM

    # Pearson correlation btw model and BOLD
    RDM_model = RDM_model_func() #(276,)
    RDM_BOLD  = distance.pdist(normalize(BOLD),'correlation')
    #print("RDM_BOLD.shape: ", RDM_BOLD.shape) #(276,)
    D,p     = spearmanr(RDM_model,RDM_BOLD)
    return D

############################### Default Paths #################################

working_dir = '/bcbl/home/public/Consciousness/stats/Data'
subj_nums = list(range(28, 37))#list(set(range(1, 37)) - set([1, 18, 25, 26]))

############## Making csv File to Label the Raw BOLD with Nilearn #############

for num in subj_nums:
    subj = 'STATSMAR_{:02d}'.format(num)
    print("Starting with subj", subj, "...")
    subj_dir = os.path.join(working_dir, subj)
    psychopy_dir = sorted(glob(os.path.join(working_dir, subj, 'psychopy', 'Learning', '*.csv')))
    nilearn_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', subj)
    #mask_dir = sorted(glob(os.path.join(working_dir, 'masks_freesurfer', subj, 'rois', 'func', 'roi*.nii.gz')))
    
    if not os.path.exists(os.path.join(os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'plot', subj))):
        os.makedirs(os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'plot', subj))
    if not os.path.exists(os.path.join(os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'SL_corr', subj))):
        os.makedirs(os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'SL_corr', subj))
    
    rsa_plot_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'plot', subj)
    sl_corr_result_dir = os.path.join(working_dir, 'pymvpa_nilearn', 'nilearn', 'SL_corr', subj)
    
    ### Paths differ according to the subject numbers
    if num <= 27:
        example_func = sorted(glob(os.path.join(working_dir, subj, 'run?', 'preprodef.feat', 'example_func.nii.gz')))
        filtered_func = sorted(glob(os.path.join(working_dir, subj, 'run?', 'preprodef.feat', 'filtered_func_data.nii.gz')))
        whole_brain_mask_dir = sorted(glob(os.path.join(working_dir, subj,'run?', 'preprodef.feat', 'mask.nii.gz')))
    
    else:
        example_func = sorted(glob(os.path.join(working_dir, subj, 'run?.feat', 'ICA-AROMA.feat', 'example_func.nii.gz')))
        filtered_func = sorted(glob(os.path.join(working_dir, subj, 'run?.feat', 'ICA-AROMA.feat', 'filtered_func_data.nii.gz')))
        whole_brain_mask_dir =  sorted(glob(os.path.join(working_dir, subj, 'run?.feat', 'ICA-AROMA.feat', 'mask.nii.gz')))  
    
    ### Load the fMRI data with nibibal
    dir_dic = {'fMRI_dir': [], 'behav_dir':[]}

    for feat, behav in zip(filtered_func, psychopy_dir):
        dir_dic['fMRI_dir'].append(feat)
        dir_dic['behav_dir'].append(behav)

    dir_df = pd.DataFrame(dir_dic)
    #print("dir_df.shape", dir_df.shape) # (8, 2)    
    

    for n_run, row in dir_df.iterrows():
        run_folder = 'run'+str(n_run + 1)
        run_dir = os.path.join(working_dir, subj, str(n_run))
        
        fMRI_dir  = row['fMRI_dir']
        # pick the one psychopy file that contains the same "session" and "block" as the MRI file
        behav_dir =  row['behav_dir']
        
        BOLD = load(fMRI_dir)
        #print("fMRI_dir")
        behav = pd.read_csv(behav_dir)

        ### Pick the HDR from 4s to 7s
        col_of_interest = ['onset_time',]
        
        ### 1.2 TR * 8 volumes to be discarded
        for col in col_of_interest:
            behav[col] -= 9.6 

        ### Pick the volumes that we are interested in between 4 to 7 sec
        behav['start'] = behav['onset_time'] + 4
        behav['end'] = behav['onset_time'] + 6

        #TR = BOLD.header['pixdim'][4] ### Actually, it's 1.2

        ### Extract the stimulus information from psychopy files
        onsets = behav['onset_time'].values
        trials = behav['trials.thisTrialN'].values
        triplet = behav['tripletsnum'].values

        total_volumes = BOLD.shape[-1] ### 344
        ### Time vector containing start times in each volume
        time_coor = np.arange(0, total_volumes*2, 2) 
        """
        NOW CALLS THE FUNCION label_trials - which labes relevant TRs with trialnumber and memory_status name
        """
        trials = np.zeros(time_coor.shape) # 1 per vol
        targets = np.array(['____no_use____'] * time_coor.shape[0])
        # print(targets)

        ### Filling in information
        for yy,row in behav.iterrows(): # 96
            """
            Pick all the rest of rows where the time coordinates are greater
            than the begining of the given trial
            say for trial 1, all the rows will be selected
            but it is fine, in trial 2, those are marked as trial 1
            will be leave alone because their time coordinates are less than
            the beginning of trial 2 and so on. - dsoto
            """
            idx                 = np.where(time_coor >= round(row['start'],0))
            trials[idx]         = yy + 1
            targets[idx]        = row['tripletsnum']

        vols = pd.DataFrame(dict(
                time_coor       = time_coor,
                trials          = trials, # to group later for averaging vols that belong to the same trial
                targets         = targets,))

        """
        df[['start','end']].values is a 344 X 2 matrix that contains all the possible time intervals.
        if any of the time in the time coordinate falls between any of the possible
        time intervals, those are what we are interested in, mark them "1" else "0"
        """
        temp = []
        for ii,row in vols.iterrows():
            time = row['time_coor']

            if any([np.logical_and(interval[0] <= time,
                               time < interval[1]) for interval in behav[['start','end']].values.round(0)]):
                temp.append(1)
            else:
                temp.append(0)
                
        ### labeling the total (344) volumes
        vols['volume_interest'] = temp
        labeling_csv_name = 'subj_{:02d}_run_{}_labeling.csv'.format(num, n_run)
        vols.to_csv(os.path.join(nilearn_dir,labeling_csv_name), index = False)

####################### Labeling BOLD Data using Nilearn ######################
        
        ### If set to False, it wont average the volumes within the 4-7s
        for average in [True]: 
            #for item,csv in zip(filtered_func, psychopy_dir):
            mask_name   = 'whole_brain_mask'#_dir.split('/')[-1].split('.')[0
            
            # do not perform any preprocessing when applying the masking
            masker      = NiftiMasker(mask_img          = os.path.abspath(whole_brain_mask_dir[n_run]), #nilearn function
                                      standardize       = False,
                                      detrend           = False,
                                      memory            = 'nilarn_cashed')
            BOLD        = masker.fit_transform(X        = fMRI_dir)
            
            df_concat   = vols
            
            idx             = df_concat['volume_interest'] == 1 # pick relevant vols from step 1
            processed_BOLD  = BOLD[idx]
            processed_df    = df_concat[idx]
            # preprocessing is applied on the picked volumes
            processed_BOLD  = clean_signal(processed_BOLD, #nilearn function
                                            t_r          = 2,
                                            detrend      = True,
                                            standardize  = True)
        
            ### Groupby vols that belong to the same trial
            if average:
                processed_BOLD,processed_df = groupby_average(processed_BOLD, 
                        processed_df.reset_index(), ['trials'])

            else:
                processed_BOLD, processed_df = processed_BOLD, processed_df
            
            BOLD = masker.inverse_transform(X = processed_BOLD) # 153608
            
            ### To slice up
            BOLD = np.array(BOLD.dataobj)
            #print("raw_BOLD_shape", BOLD.shape)
            
############### Sorting Npy Values after Loading with Nilearn #################
            
            ## Sorting stimuli on the dataframe
            new_triple_num_li = []
            triple_char_li = []
            triple_group_li = []
    
            for target in processed_df['targets'].astype(int):
                 #print("target: ", target, "new_triple_num: ", triple_num_dic[target])
                new_triple_num = triple_num_dic[target]
                triple_char = triple_char_dic[target]
                triple_group = triple_group_dic[target]
                
                new_triple_num_li.append(new_triple_num)
                triple_char_li.append(triple_char)
                triple_group_li.append(triple_group)
            
            processed_df['new_triple_num'] = new_triple_num_li
            processed_df['triple_char'] = triple_char_li
            processed_df['triple_group'] = triple_group_li
            
            ### Sorting npy according to the target
            processed_df = processed_df.sort_values(['triple_group', 'trials'], ascending=[True, True])
            arg_sort_idx = processed_df.index.values
            #print("processed_df", processed_df.head())
            #print("argsort_idx: ", len(arg_sort_idx), arg_sort_idx)
                
            sorted_BOLD = processed_BOLD[np.argsort(arg_sort_idx)]
            #print("sorted_BOLD", sorted_BOLD.shape)
            
            ### Combining df and BOLD for Later Use
            cols = processed_df.columns.to_list()
            for i in list(range(sorted_BOLD.shape[1])):
                cols.append(i)
            df = pd.DataFrame(np.column_stack([processed_df, sorted_BOLD]), columns = cols)
            
            ### Averaging according to the unique stimuli
            num_cols = df.columns[pd.to_numeric(df.columns, errors='coerce').to_series().notnull()]
            df_mean = df.groupby(by='triple_char', as_index=False)[num_cols].apply(lambda x:x.mean())

            total_BOLD_mean_li = []
            for unique_target in list(string.ascii_lowercase)[:24]:
                #print("unique_target", unique_target)
                group_by_char_idx = processed_df.loc[processed_df['triple_char']==unique_target].index.values
                #print("group_by_char_idx", group_by_char_idx)
                BOLD_by_char = BOLD[:, :, :, group_by_char_idx]
                BOLD_by_char_mean = np.mean(BOLD_by_char, axis=3)
                #print("BOLD_by_char_mean.shape", BOLD_by_char_mean.shape)
                total_BOLD_mean_li.append(BOLD_by_char_mean)
            avg_BOLD = np.stack(total_BOLD_mean_li, axis=3)

            ### Saving
            avg_BOLD_filename = 'avg_{}_BOLD.npy'.format(mask_name)
            np.save(os.path.join(nilearn_dir, str(n_run), avg_BOLD_filename), avg_BOLD,)
            df_mean.to_csv(os.path.join(nilearn_dir, str(n_run),'avg_{}_events.csv'.format(mask_name)),index = False)

            radius = 6 # in mm
            
            import time
            begin_time = time.time()
            
            ### Searchlight by Brainiak
            sl = Searchlight(sl_rad = radius, 
                             max_blk_edge = radius - 1, 
                             shape = Ball, 
                             min_active_voxels_proportion = 0,)
            """
            Distributes the data based on the sphere.
            The first input is usually the BOLD signal, and it is in the form of 
            lists not arrays, representing each subject.
            The second input is usually the mask, and it is in the form of array
            """
            ### Distribute the brain_data to SearchLight to prepare it to run.
            sl.distribute([avg_BOLD],np.asanyarray(load(whole_brain_mask_dir[n_run]).dataobj) == 1)
            ### Data needed for searchlight is sent to all the cores.
            sl.broadcast(model)
            ### Run SearchLight
            global_outputs_no_tune = sl.run_searchlight(sfn, pool_size = -1,) # a single CPU
            
            end_time = time.time()
            
            ### Convert it into 4D img # let's try with example_func instead of filtered func
            correlations_no_tune = new_img_like(example_func[n_run], np.asanyarray(global_outputs_no_tune,dtype = np.float32))
            correlations_no_tune = masker.inverse_transform(masker.transform_single_imgs(correlations_no_tune))
            
            print('It took {:.2f} s'.format(end_time - begin_time), 'for subj_{} ,run{}'.format(subj, n_run))
            ### Save it into a npy file
            #global_outputs_no_tune[np.isnan(global_outputs_no_tune)] = 0 #TypeError: ufunc 'isnan' not supported for the input types,
            np.save(os.path.join(sl_corr_result_dir, 'sl_result_of_run{}'.format(n_run)), global_outputs_no_tune)
            
            ### Plotting
            plotting.plot_stat_map(correlations_no_tune,
                                   example_func[n_run],
                       threshold = 1e-3,
                       draw_cross = False,
                       cmap = plt.cm.coolwarm,
                       vmax = .1,
                       title = 'Search Light for '+subj+' '+run_folder,
                       cut_coords = [0,0,25],)

            plt.savefig(rsa_plot_dir+"/"+'Search Light for '+subj+' '+run_folder+".png")
