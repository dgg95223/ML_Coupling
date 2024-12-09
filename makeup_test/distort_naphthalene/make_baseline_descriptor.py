import numpy as np
import subprocess
import sys, copy
import ML_Coupling.mo_descriptor2 as md

'''
prepare data_set
1. make baseline descriptor
baseline model for translation along x, y, z axis, no rotation is applied
'''
BOHR = 0.52917721092
# trans
results_csv_path = '/home/jingheng/ML_Coupling/data/results.csv'
# rot
results_csv_path = '/home/jingheng/ML_Coupling/data/results_rot_ss_2.csv'

# trans only
# d_trans = np.loadtxt(results_csv_path,comments='#', usecols=(0,1,2), delimiter=',') * 0.1
# d_rot   = np.zeros(d_trans.shape)
# d_rot[:,0] = np.ones(len(d_trans)) * 10

# rot only
d_rot   = np.loadtxt(results_csv_path,comments='#', usecols=(0,1,2), delimiter=',')
d_trans = np.zeros(d_rot.shape)
d_trans[:,2] = np.ones(len(d_trans)) * 4

des = np.array([(d_trans[i], d_rot[i]) for i in range(0,len(d_trans))])

# print(des[0:10])

np.save('../../data/baseline_rot.npy', des)
# homo_pairs = np.load('../data/homo_homo_pair2.npy')