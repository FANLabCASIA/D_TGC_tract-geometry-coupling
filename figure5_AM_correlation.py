import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import sparse
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from scipy.stats import pearsonr

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings('ignore')

############################################# config #############################################
# config_parser = parser = argparse.ArgumentParser()
# parser.add_argument('--tasknamei', type=int, default='0', help='the index of activation map')

# def _parse_args():
#     # The main arg parser parses the rest of the args, the usual
#     # defaults will have been overridden if config file specified.
#     args = parser.parse_args()

#     # Cache the args as a text string to save them in the output dir later
#     args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
#     return args, args_text

# args, args_text = _parse_args()
# print(args)

# # config
# tasknamei = args.tasknamei

############################################# basic info #############################################
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

# the medial wall
dirc_L = '/n02dat01/users/dyli/Atlas/metric_index_L.txt'
select_ind_L = np.loadtxt( dirc_L ).astype(int)
dirc_R = '/n02dat01/users/dyli/Atlas/metric_index_R.txt'
select_ind_R = np.loadtxt( dirc_R ).astype(int)

TaskList = ['EMOTION-cope1','EMOTION-cope2','EMOTION-cope3',
            'GAMBLING-cope1','GAMBLING-cope2','GAMBLING-cope3',
            'LANGUAGE-cope1','LANGUAGE-cope2','LANGUAGE-cope4',
            'MOTOR-cope1','MOTOR-cope2','MOTOR-cope3','MOTOR-cope4','MOTOR-cope5','MOTOR-cope6','MOTOR-cope7','MOTOR-cope8','MOTOR-cope9','MOTOR-cope10','MOTOR-cope11','MOTOR-cope12','MOTOR-cope13',
            'RELATIONAL-cope1','RELATIONAL-cope2','RELATIONAL-cope4',
            'SOCIAL-cope1','SOCIAL-cope2','SOCIAL-cope6',
            'WM-cope1','WM-cope2','WM-cope3','WM-cope4','WM-cope5','WM-cope6','WM-cope7','WM-cope8','WM-cope9',
            'WM-cope10','WM-cope11','WM-cope15','WM-cope16','WM-cope17','WM-cope18','WM-cope19','WM-cope20','WM-cope21','WM-cope22',
            ]

TaskName = ['EMOTION-FACES','EMOTION-SHAPES','EMOTION-FACES-SHAPES',
            'GAMBLING-PUNISH','GAMBLING-REWARD','GAMBLING-PUNISH-REWARD',
            'LANGUAGE-MATH','LANGUAGE-STORY','LANGUAGE-STORY-MATH',
            'MOTOR-CUE','MOTOR-LF','MOTOR-LH','MOTOR-RF','MOTOR-RH','MOTOR-T','MOTOR-AVG','MOTOR-CUE-AVG','MOTOR-LF-AVG','MOTOR-LH-AVG','MOTOR-RF-AVG','MOTOR-RH-AVG','MOTOR-T-AVG',
            'RELATIONAL-MATCH','RELATIONAL-REL','RELATIONAL-REL-MATCH',
            'SOCIAL-RANDOM','SOCIAL-TOM','SOCIAL-TOM-RANDOM',
            'WM-2BK_BODY','WM-2BK_FACE','WM-2BK_PLACE','WM-2BK_TOOL','WM-0BK_BODY','WM-0BK_FACE','WM-0BK_PLACE','WM-0BK_TOOL','WM-2BK',
            'WM-0BK','WM-2BK-0BK','WM-BODY','WM-FACE','WM-PLACE','WM-TOOL','WM-BODY-AVG','WM-FACE-AVG','WM-PLACE-AVE','WM-TOOL-AVE',
            ]

namelist = [
 '100307', '100408', '101107', '101309', '101915', '103111', '103414', '103818', '105014', '105115', 
 '106016', '108828', '110411', '111312', '111716', '113619', '113922', '114419', '115320', '116524', 
 '117122', '118528', '118730', '118932', '120111', '122317', '122620', '123117', '123925', '124422', 
 '125525', '126325', '127630', '127933', '128127', '128632', '129028', '130013', '130316', '131217', 
 '131722', '133019', '133928', '135225', '135932', '136833', '138534', '139637', '140925', '144832', 
 '146432', '147737', '148335', '148840', '149337', '149539', '149741', '151223', '151526', '151627', 
 '153025', '154734', '156637', '159340', '160123', '161731', '162733', '163129', '176542', '178950', 
 '188347', '189450', '190031', '192540', '196750', '198451', '199655', '201111', '208226', '211417', 
 '211720', '212318', '214423', '221319', '239944', '245333', '280739', '298051', '366446', '397760', 
 '414229', '499566', '654754', '672756', '751348', '756055', '792564', '856766', '857263', '899885', 
 ]

def GAMMA_my(data):
    from sklearn.mixture import GaussianMixture
    from scipy.stats import gamma

    # 使用 GaussianMixture 拟合数据
    gmm = GaussianMixture(n_components=2)  # 一个高斯分布和一个Gamma分布
    gmm.fit(data.reshape(-1, 1))

    # 获取每个分布的均值和方差
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    print(means, variances)

    # 正激活阈值和负激活阈值分别设定为两个Gamma分布的中位数
    gamma1_mean = means[variances.argmax()]

    # 对于Gamma分布，中位数等于shape参数乘以尺度参数的自然对数
    gamma_threshold = gamma.ppf(0.5, a=2, scale=gamma1_mean/2)

    return gamma_threshold

def normalization_my(x:np.ndarray):
    # x: (29696,)
    x_pos = x.copy()
    x_pos[x<0] = 0
    x_pos = x_pos / np.max(x_pos)

    x_neg = x.copy()
    x_neg[x>0] = 0
    x_neg = -1 *x_neg / np.min(x_neg)

    return x_neg + x_pos

############################################# main function #############################################
for tasknamei in range(47):
    taskname = TaskList[tasknamei]
    print(taskname)
    activation_name = taskname.split('-')[0]
    cope_name = taskname.split('-')[1]

    if not os.path.exists(f'/n01dat01/dyli/multi/results_data/AM_prediction_correlation/correlationP_{taskname}.txt'):
        # y shape: (100, 29696); x shape: (100, 7164)
        y_ori = np.loadtxt(f'/n01dat01/dyli/multi/results_data/AM_prediction_expression/y_{taskname}_ori.txt')
        y_thr = np.loadtxt(f'/n01dat01/dyli/multi/results_data/AM_prediction_expression/y_{taskname}_thrGAMMA.txt')
        x     = np.loadtxt(f'/n01dat01/dyli/multi/results_data/AM_prediction_expression/x_ori.txt')

        def para_calculate_my(i):
            y = y_ori[:, i] # y: (100,1)
            result = np.zeros(x.shape[1])
            for v in range(x.shape[1]):
                r,result[v] = pearsonr(y, np.squeeze(x[:,v]))
            return result

        inputs = range(y_ori.shape[1])
        num_cores = multiprocessing.cpu_count()
        print('the number of cores: ', num_cores)
        para_results = Parallel(n_jobs=num_cores)(delayed(para_calculate_my)(i) for i in inputs)
        para_results = np.squeeze(np.array(para_results))
        print('the size of parameter results: ', para_results.shape)

        np.savetxt(f'/n01dat01/dyli/multi/results_data/AM_prediction_correlation/correlationP_{taskname}.txt', para_results)
        print('Finished!!')