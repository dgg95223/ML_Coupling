import tensorflow as tf
import numpy as np
import subprocess
import sys, copy
subprocess.run('export TF_INTRA_OP_PARALLELISM_THREADS=52', shell=True)

model_path = sys.argv[1]
print(model_path)
'''
1. load mo_pair descriptor
'''

homo_pairs = np.load('./data/homo_homo_pair.npy')
lumo_pairs = np.load('./data/lumo_lumo_pair.npy')
homo_rot_pairs = np.load('./data/homo_pair_rot.npy')
lumo_rot_pairs = np.load('./data/lumo_pair_rot.npy')

homo_pairs_tr = np.concatenate((homo_pairs,homo_rot_pairs))
lumo_pairs_tr = np.concatenate((lumo_pairs,lumo_rot_pairs))

'''
2. read coupling
'''
raw_data = np.loadtxt('./data/results.csv', delimiter=',',comments='#')
raw_data_rot = np.loadtxt('./data/results_rot.csv', delimiter=',',comments='#')
c_homo = abs(raw_data[:,3])
c_lumo = abs(raw_data[:,4])
c_homo_rot = abs(raw_data_rot[:,3])
c_lumo_rot = abs(raw_data_rot[:,4])

c_homo_tr = np.concatenate((c_homo,c_homo_rot))
c_lumo_tr = np.concatenate((c_lumo,c_lumo_rot))

'''
3. remove zero values
'''
ihzero = []
for ii,i in np.ndenumerate(c_homo):
    if i<=0.00000000:
        ihzero.append(ii)
        c_homo[ii] = 1e-9
print('Number of points to be deleted for homo:  ',len(ihzero))
homo_pairs  = np.delete(homo_pairs , ihzero, 0)
c_homo  = np.delete(c_homo , ihzero, 0)

ilzero = []
for ii,i in np.ndenumerate(c_lumo_tr):
    if i<=0.00000000:
        ilzero.append(ii)
        c_lumo_tr[ii] = 1e-9
print('Number of points to be deleted for lumo:  ',len(ilzero))
lumo_pairs_tr  = np.delete(lumo_pairs_tr , ilzero, 0)
c_lumo_tr  = np.delete(c_lumo_tr , ilzero, 0)

'''
4. build training set
'''
train_homo_pairs = homo_pairs[:]
train_lumo_pairs = lumo_pairs_tr[:]

train_c_homo = -np.log(c_homo)[:]
train_c_lumo = -np.log(c_lumo_tr)[:]

train_homo = copy.deepcopy(train_homo_pairs)
train_chomo = copy.deepcopy(train_c_homo)
print('Size of full training set for homo:   ',len(train_chomo))
index = np.random.choice(len(train_c_homo), size=int(len(train_c_homo)*0.1), replace=False)
train_homo_ = np.delete(train_homo,index,0)
train_chomo_ = np.delete(train_chomo,index,0)
print('Size of selected training set for homo:   ',len(train_homo_))

train_lumo = copy.deepcopy(train_lumo_pairs)
train_clumo = copy.deepcopy(train_c_lumo)
print('Size of full training set for lumo:   ',len(train_clumo))
index = np.random.choice(len(train_c_lumo), size=int(len(train_c_lumo)*0.1), replace=False)
train_lumo_ = np.delete(train_lumo,index,0)
train_clumo_ = np.delete(train_clumo,index,0)
print('Size of selected training set for lumo:   ',len(train_lumo_))

'''
5. build testing set
'''
iall = np.arange(len(train_c_homo))
idiff = np.setdiff1d(iall,index)
test_homo = np.delete(copy.deepcopy(train_homo_pairs),idiff,0)
test_chomo = np.delete(copy.deepcopy(train_chomo),idiff,0)

iall = np.arange(len(train_c_lumo))
idiff = np.setdiff1d(iall,index)
test_lumo = np.delete(copy.deepcopy(train_lumo_pairs),idiff,0)
test_clumo = np.delete(copy.deepcopy(train_clumo),idiff,0)

'''
6. load model
'''
model = tf.keras.models.load_model(model_path, compile=False)
pred = np.exp(-model(homo_pairs, training=False).numpy().reshape((len(homo_pairs),)))

print(np.mean(np.multiply(abs(np.log(pred)-np.log(c_homo)), np.power(-np.log(c_homo),-1))*100))
error1 = np.mean(np.multiply(abs(pred-c_homo), np.power(c_homo,-1))*100)
error2 = np.mean(abs(pred-c_homo)*27.2114*1000)
print('MAPE: ',error1, '\nMAE(meV): ',error2)
