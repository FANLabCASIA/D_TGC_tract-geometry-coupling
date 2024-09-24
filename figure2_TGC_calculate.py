import os
import sys
import yaml
import argparse
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import sparse
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from nilearn import image, surface, plotting, datasets
from tqdm import tqdm
from matplotlib import font_manager
font_manager.fontManager.addfont("/n02dat01/users/lchai/anaconda3/envs/Nm/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/arial.ttf")
plt.rcParams["font.sans-serif"] = "Arial" 

import warnings
warnings.filterwarnings('ignore')

############################################# config #############################################
config_parser = parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=str, default='100307', help='')

def _parse_args():
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

args, args_text = _parse_args()
print(args)

# config
sub = args.sub
############################################# config #############################################

# remove the medial wall
dirc_L = '/n02dat01/users/dyli/Atlas/metric_index_L.txt'
select_ind_L = np.loadtxt( dirc_L ).astype(int)
dirc_R = '/n02dat01/users/dyli/Atlas/metric_index_R.txt'
select_ind_R = np.loadtxt( dirc_R ).astype(int)

# fiber index
l_idx = [1,3,13,15,17,19,21,23,25,27,30,32,34,36,38,40,42,44,47,49,51,53,55,57,59,61,63,65,67,69,71]
r_idx = [2,4,14,16,18,20,22,24,26,28,31,33,35,37,39,41,43,45,48,50,52,54,56,58,60,62,64,66,68,70,72]
m_idx = [4,5,6,7,8,9,10,11,28,45]
l_idx = np.array(l_idx)
r_idx = np.array(r_idx)
m_idx = np.array(m_idx)
l_idx = l_idx-1
r_idx = r_idx-1

# read the group MODE
x = np.loadtxt('/n01dat01/dyli/multi/support_code/BrainEigenmodes/data/template_eigenmodes/fsLR_32k_white-lh_emode_200.txt') # (32492, 200)
x = x[select_ind_L,:]


if not os.path.exists(f'/n01dat01/dyli/multi/HCP_1200/{sub}'): os.mkdir(f'/n01dat01/dyli/multi/HCP_1200/{sub}')
if not os.path.exists(f'/n01dat01/dyli/multi/HCP_1200/{sub}/FP_{sub}_predict_by_200_group_whitemode_thr05_para_L.npy'):
    # read the fingerprint
    _ = sps.load_npz(f'/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas/{sub}/{sub}_L_probtrackx_omatrix2/finger_print_fiber_MSMALL.npz')
    _ = _.toarray()

    # check the all-zero row
    for ii in range(_.shape[0]):
        if len(np.unique(_[ii,:])) == 1: _[ii,:]=_[ii-1,:]
    for ii in range(_.shape[0]):
        if len(np.unique(_[ii,:])) == 1: sys.exit()

    # normalization
    fingerprint = np.array([_[:,i]/np.sum(_, axis=1) for i in range(_.shape[1])]).T

    # choose the left fibers and the cc fibers
    fingerprint = fingerprint[:, np.array(list(l_idx) + list(m_idx))]
    assert fingerprint.shape[0]==29696 and fingerprint.shape[1]==int(len(l_idx)+len(m_idx))

    # thr
    fingerprint[fingerprint<0.05] =0

    # GLM model
    corr_re = np.zeros(fingerprint.shape[1])
    para = np.zeros((200, fingerprint.shape[1]))
    for ff in range(fingerprint.shape[1]):
        y = fingerprint[:, ff]
        glm = sm.GLM(y,x, family=sm.families.Gaussian())
        glm_results = glm.fit()
        para[:,ff] = glm_results.params.T
        corr_re[ff] = np.corrcoef(np.squeeze(y), np.squeeze(np.dot(x, glm_results.params.T)))[0,1]
    
    # save the correlation results for each sub
    np.save(f'/n01dat01/dyli/multi/HCP_1200/{sub}/FP_{sub}_predict_by_200_group_whitemode_deve-mode_thr05_para_L.npy', para)
    np.save(f'/n01dat01/dyli/multi/HCP_1200/{sub}/FP_{sub}_predict_by_200_group_whitemode_deve-mode_thr05_pearsonr_L.npy', corr_re)