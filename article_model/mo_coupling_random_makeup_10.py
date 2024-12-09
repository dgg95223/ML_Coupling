import ML_Coupling.nn_frame as nn
import numpy as np
import subprocess
import copy,glob
import tensorflow as tf

subprocess.run('export TF_INTRA_OP_PARALLELISM_THREADS=12', shell=True)

'''
1&2. load mo_pair descriptor and coupling
'''
raw_data = np.loadtxt('../data/results.csv', delimiter=',',comments='#')
c_homo = abs(raw_data[:,3]) 
c_lumo = abs(raw_data[:,4])

homo_pairs = np.load('../data/homo_homo_pair2.npy')
lumo_pairs = np.load('../data/lumo_lumo_pair2.npy')

'''
4. remove zero values for full data set
'''
ihzero = []
for ii,i in np.ndenumerate(c_homo):
    if i<=0.000001:
        ihzero.append(ii)
        c_homo[ii] = 1e-6
print('Number of points to be deleted for homo:  ',len(ihzero))
homo_pairs  = np.delete(homo_pairs , ihzero, 0)
c_homo= np.delete(c_homo , ihzero, 0)

ilzero = []
for ii,i in np.ndenumerate(c_lumo):
    if i<=0.000001:
        ilzero.append(ii)
        c_lumo[ii] = 1e-6
print('Number of points to be deleted for lumo:  ',len(ilzero))
lumo_pairs  = np.delete(lumo_pairs , ilzero, 0)
c_lumo = np.delete(c_lumo , ilzero, 0)

'''
6&7. build training set and testing set
'''
train_homo_pairs = homo_pairs[:]
train_lumo_pairs = lumo_pairs[:]

train_c_homo = -np.log(c_homo)[:]
train_c_lumo = -np.log(c_lumo)[:]
    
'''
8. load model and train, test
''' 
data_type = 'trans'
model_type = 'homo'
model_paths = glob.glob('../article_model/'+data_type+'/nat*'+model_type+'*60430')[0]
model = tf.keras.models.load_model(model_paths+'/model/', compile=False)
pred_ho_1 = np.exp(-model(homo_pairs, training=False).numpy().reshape((len(homo_pairs),)))

print(model_paths)
error1 = np.mean(np.multiply(abs(pred_ho_1-c_homo), np.power(c_homo,-1))*100)
error2 = np.mean(abs(pred_ho_1-c_homo)*1000 * 27.2114)
print('Model path: ', model_paths)
print('MAPE: ',error1, '\nMAE(meV): ',error2)