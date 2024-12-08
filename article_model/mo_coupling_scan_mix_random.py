import ML_Coupling.nn_frame as nn
import numpy as np
import subprocess,glob
import copy
import tensorflow as tf
subprocess.run('export TF_INTRA_OP_PARALLELISM_THREADS=52', shell=True)


'''
1&2. load mo_pair descriptor and coupling
'''
raw_data = np.loadtxt('../data/results.csv', delimiter=',',comments='#')
raw_data_rot = np.loadtxt('../data/results_rot_ss_2.csv', delimiter=',',comments='#')
c_homo = abs(raw_data[:,3])
c_lumo = abs(raw_data[:,4])
c_homo_rot = abs(raw_data_rot[:,3])
c_lumo_rot = abs(raw_data_rot[:,4])
close = raw_data_rot[:,6]

homo_pairs = np.load('../data/homo_homo_pair2.npy')
lumo_pairs = np.load('../data/lumo_lumo_pair2.npy')
homo_pairs_rot = np.load('../data/homo_pair_rot2.npy')
lumo_pairs_rot = np.load('../data/lumo_pair_rot2.npy')
'''
4. remove zero values for full data set
'''
iclose = []
for ii,i in np.ndenumerate(close):
    if i == 1:
        iclose.append(ii)
print('Number of points to be deleted for close configuratiion:  ',len(iclose))

homo_pairs_rot  = np.delete(homo_pairs_rot , iclose, 0)
lumo_pairs_rot  = np.delete(lumo_pairs_rot , iclose, 0)
c_homo_rot= np.delete(c_homo_rot , iclose, 0)
c_lumo_rot= np.delete(c_lumo_rot , iclose, 0)

'''
5. even the size of translation and rotation part and remove zero values
'''
homo_pairs_ = copy.deepcopy(homo_pairs)
c_homo_ = copy.deepcopy(c_homo)
index = np.random.choice(len(c_homo), size=int(len(c_homo)*0.75), replace=False)
_homo_pairs = np.delete(homo_pairs_,index,0)
_c_homo = np.delete(c_homo_,index,0)

lumo_pairs_ = copy.deepcopy(lumo_pairs)
c_lumo_ = copy.deepcopy(c_lumo)
index = np.random.choice(len(c_lumo), size=int(len(c_lumo)*0.75), replace=False)
_lumo_pairs = np.delete(lumo_pairs_,index,0)
_c_lumo = np.delete(c_lumo_,index,0)

c_homo_tr = np.concatenate((_c_homo,c_homo_rot))
c_lumo_tr = np.concatenate((_c_lumo,c_lumo_rot))
homo_pairs_tr = np.concatenate((_homo_pairs,homo_pairs_rot))
lumo_pairs_tr = np.concatenate((_lumo_pairs,lumo_pairs_rot))

homo_pairs = homo_pairs_tr
lumo_pairs = lumo_pairs_tr
c_homo = c_homo_tr    
c_lumo = c_lumo_tr

ihzero = []
for ii,i in np.ndenumerate(c_homo):
    if i<=0.000001:
        ihzero.append(ii)
print('Number of points to be deleted for homo:  ',len(ihzero))
homo_pairs  = np.delete(homo_pairs , ihzero, 0)
c_homo= np.delete(c_homo , ihzero, 0)

ilzero = []
for ii,i in np.ndenumerate(c_lumo):
    if i<=0.000001:
        ilzero.append(ii)
print('Number of points to be deleted for lumo:  ',len(ilzero))
lumo_pairs  = np.delete(lumo_pairs , ilzero, 0)
c_lumo = np.delete(c_lumo , ilzero, 0)

'''
6&7. build training set and testing set
'''
data_type = 'mix2'
model_type = 'homo'
model_paths = glob.glob('../article_model/'+data_type+'/nat*'+model_type+'*')[5]
model = tf.keras.models.load_model(model_paths+'/model/', compile=False)
pred_ho_1 = np.exp(-model(homo_pairs, training=False).numpy().reshape((len(homo_pairs),)))

print(model_paths)
error1 = np.mean(np.multiply(abs(pred_ho_1-c_homo), np.power(c_homo,-1))*100)
error2 = np.mean(abs(pred_ho_1-c_homo)*1000 * 27.2114)
print('Model path: ', model_paths)
print('MAPE: ',error1, '\nMAE(meV): ',error2)
    
