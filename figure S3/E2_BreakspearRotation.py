import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import sparse
import scipy as sp
import statsmodels.api as sm
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings('ignore')


# remove the medial wall
dirc_L = '/n02dat01/users/dyli/Atlas/metric_index_L.txt'
select_ind_L = np.loadtxt( dirc_L ).astype(int)
dirc_R = '/n02dat01/users/dyli/Atlas/metric_index_R.txt'
select_ind_R = np.loadtxt( dirc_R ).astype(int)

_ = np.array([0, 2, 12, 14, 16, 18, 20, 22, 24, 26, 29, 31, 33, 35, 37, 39, 41, 43, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 4, 5, 6, 7, 8, 9, 10, 11, 28, 45])
# read the fiber name
l_idx = [1,3,13,15,17,19,25,27,30,32,36,38,40,42,44,47,49,51,53,55,57,59,61,63,65,67,69,71]
r_idx = [2,4,14,16,18,24,26,28,31,35,37,39,41,43,45,48,50,52,54,56,58,60,62,64,66,68,70,72]
m_idx = [4,5,6,7,8,9,10,11]
l_idx = np.array(l_idx)
r_idx = np.array(r_idx)
m_idx = np.array(m_idx)
l_idx = l_idx-1
r_idx = r_idx-1
label_f = open('/n02dat01/users/dyli/Grad_data/support_data/fiber_name_ori_nonum_nohemi.txt', 'r')
label_name = label_f.readlines()
label_name = [' '.join([i.strip() for i in price.strip().split('\n')]) for price in label_name]
label_name_lm = [label_name[l_idx[i]] for i in range(len(l_idx))] + [label_name[m_idx[i]] for i in range(len(m_idx))]
print(f'the number of fiber: {len(label_name_lm)}')

new_fiber_idx = []
for fi,ff in enumerate(_):
    if ff in list(l_idx)+list(m_idx): new_fiber_idx.append(fi)
new_fiber_idx = np.array(new_fiber_idx)
print(new_fiber_idx.shape)

print(len(l_idx), len(m_idx))

# read the sublist
list_path = '/n02dat01/users/dyli/Grad_data/support_data/HCP_U100_list.txt'
with open( list_path, 'r' ) as f:
    namelist = [ str( line.strip()) for line in f.readlines() ]
print(f'the sub num is {len(namelist)}')

print('read sur modes')
M_sur = np.load('/n01dat01/dyli/multi/results_data/RotationTest_Breakspear/fsLR_32k_white-lh_emode_200_Sur1000.npy')
print('finish reading')

for sub in namelist[0:50]:
    print(sub)
    if not os.path.exists(f'/n01dat01/dyli/multi/results_data/RotationTest_Breakspear/HCP_U100/FP_{sub}_predict_by_200_group_whitemode_deve-mode_thr05_para_L.npy'):
        # read the fingerprint
        if os.path.exists(f'/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas/{sub}/{sub}_L_probtrackx_omatrix2/finger_print_fiber_MSMALL.npz'):
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

        # for each item
        def para_calculate_my(i):
            x = np.squeeze(M_sur[...,i])
            x = x[select_ind_L,:]

            corr_re = np.zeros(fingerprint.shape[1])
            para = np.zeros((200, fingerprint.shape[1]))
            for ff in range(fingerprint.shape[1]):
                y = fingerprint[:, ff]
                glm = sm.GLM(y,x, family=sm.families.Gaussian())
                glm_results = glm.fit()
                para[:,ff] = glm_results.params.T
                corr_re[ff] = np.corrcoef(np.squeeze(y), np.squeeze(np.dot(x, glm_results.params.T)))[0,1]

            assert ~np.isnan(para).any()
            assert ~np.isnan(corr_re).any()
            return para

        inputs = range(1000)
        num_cores = multiprocessing.cpu_count()
        print('the number of cores: ', num_cores)
        para_results = Parallel(n_jobs=num_cores)(delayed(para_calculate_my)(i) for i in inputs)
        para_results = np.squeeze(np.array(para_results))
        
        # save the correlation results for each sub
        np.save(f'/n01dat01/dyli/multi/results_data/RotationTest_Breakspear/HCP_U100/FP_{sub}_predict_by_200_group_whitemode_deve-mode_thr05_para_L.npy', para_results)
        print('-'*15,'Finished!!', '-'*15)
    else:
        print('-'*15,'Finished!!', '-'*15)
        