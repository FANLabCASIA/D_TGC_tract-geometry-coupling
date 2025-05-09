import os
import sys
import yaml
import argparse
from tqdm import tqdm
import numpy as np

from scipy.sparse.linalg import eigsh, eigs

from brainsmash.mapgen.sampled import Sampled
from brainsmash.mapgen.memmap import txt2memmap

############################################# config #############################################
config_parser = parser = argparse.ArgumentParser()
parser.add_argument('--modei', type=int, default=0, help='')

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
modei = args.modei
modei = str(int(modei))
############################################# config #############################################

if not os.path.exists(f'{Sur_father}/surrogates_for_mode{modei}_{Sur_num}_{hemi}.npy'):
    Sur_num = 1000
    hemi = 'L'
    GeoDist_hemi = 'Left'
    # Sur_father = f'/n01dat01/dyli/multi/results_data/RotationTest_Brainsmesh'
    Sur_father = f'/n01dat01/deyingli/TGC_proj/ForAZ/RotationTest_Brainsmesh'
    if not os.path.exists(Sur_father): os.mkdir(Sur_father)

    print('='*15, 'Sur num:', Sur_num, 'Hemi:', hemi, 'result path:', Sur_father, '='*15)

    dist_mat_fin = f'/n02dat01/users/dyli/Grad_data/support_data/{GeoDist_hemi}DenseGeodesicDistmat.txt'  # input text file
    # output_dir = Sur_father  # directory to which output binaries are written
    # output_files = txt2memmap(dist_mat_fin, Sur_father, maskfile=None, delimiter=' ')

    output_files = {'distmat': f'{Sur_father}/distmat.npy',
                    'index': f'{Sur_father}/index.npy'}

    # distmat: (32492, 32492) index: (32492, 32492)

    dirc = f'/n02dat01/users/dyli/Atlas/metric_index_{hemi}.txt'
    select_ind = np.loadtxt( dirc ).astype(int)

    x = np.loadtxt(f'/n01dat01/dyli/multi/support_code/BrainEigenmodes/data/template_eigenmodes/fsLR_32k_white-lh_emode_200.txt') # (32492, 200)

    # for modei in tqdm(range(200)):
    print(modei)
    Grad = np.zeros(32492)
    Grad[select_ind] = x[select_ind,:][:,modei]
    np.savetxt(f'{Sur_father}/mode{modei}_{hemi}.txt', X=Grad, delimiter=' ')

    brain_map_file = f'{Sur_father}/mode{modei}_{hemi}.txt'
    dist_mat_mmap = output_files['distmat']
    index_mmap = output_files['index']
    sampled = Sampled(brain_map_file, dist_mat_mmap, index_mmap)

    surrogates = sampled(n=1000) # (10000, 32492)
    np.save(f'{Sur_father}/surrogates_for_mode{modei}_{Sur_num}_{hemi}', surrogates)
    print(modei, 'finished')
else:
    print(modei, 'finished')