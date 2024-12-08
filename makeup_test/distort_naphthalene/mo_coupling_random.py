import ML_Coupling.nn_frame as nn
import numpy as np
import subprocess
import copy
subprocess.run('export TF_INTRA_OP_PARALLELISM_THREADS=96', shell=True)


ratio = [0.9, 0.5, 0.1]
'''
1&2. load mo_pair descriptor and coupling
'''
raw_data = np.loadtxt('../../data/results.csv', delimiter=',',comments='#')
c_homo = abs(raw_data[:,3]) 
c_lumo = abs(raw_data[:,4])

homo_pairs = np.load('../../data/baseline.npy')
lumo_pairs = np.load('../../data/baseline.npy')

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
6$7. build training set and testing set
'''
train_homo_pairs = homo_pairs[:]
train_lumo_pairs = lumo_pairs[:]

train_c_homo = -np.log(c_homo)[:]
train_c_lumo = -np.log(c_lumo)[:]

for i in ratio:
    train_homo = copy.deepcopy(train_homo_pairs)
    train_chomo = copy.deepcopy(train_c_homo)
    print('Size of full training set for homo:   ',len(train_chomo))
    index = np.random.choice(len(train_c_homo), size=int(len(train_c_homo)*i), replace=False)
    train_homo_ = np.delete(train_homo,index,0)
    train_chomo_ = np.delete(train_chomo,index,0)
    print('Size of selected training set for homo:   ',len(train_homo_))
    
    iall = np.arange(len(train_c_homo))
    idiff = np.setdiff1d(iall,index)
    test_homo = np.delete(copy.deepcopy(train_homo_pairs),idiff,0)
    test_chomo = np.delete(copy.deepcopy(train_chomo),idiff,0)
    
    train_lumo = copy.deepcopy(train_lumo_pairs)
    train_clumo = copy.deepcopy(train_c_lumo)
    print('Size of full training set for lumo:   ',len(train_clumo))
    index = np.random.choice(len(train_c_lumo), size=int(len(train_c_lumo)*i), replace=False)
    train_lumo_ = np.delete(train_lumo,index,0)
    train_clumo_ = np.delete(train_clumo,index,0)
    print('Size of selected training set for lumo:   ',len(train_lumo_))
    
    iall = np.arange(len(train_c_lumo))
    idiff = np.setdiff1d(iall,index)
    test_lumo = np.delete(copy.deepcopy(train_lumo_pairs),idiff,0)
    test_clumo = np.delete(copy.deepcopy(train_clumo),idiff,0)
    
    '''
    8. load model and train, test
    ''' 
    setting = {'activation':'tanh','nn_shape':(256,256,256),'batch_size':len(train_homo_), 'training_steps':80000,\
    'learning_rate': 0.0008*(1-i), 'decay_rate':0.95, 'decay_per_steps':1000, 'save_step':1000, 'drop_rate':0, 'save_path':'./trained_model/trans/nat_dimer_baseline_trans_%d'%len(train_homo_),\
    'seed':None, 'debug_traj':False, 'pre_trained_path':None, 'baseline': True}
    NN_ho = nn.NN(setting_dict=setting)
    NN_ho.train(train_homo_,train_chomo_)
    pred_ho_1 = np.exp(-NN_ho.model(homo_pairs, training=False).numpy().reshape((len(homo_pairs),)))
    pred_ho_2 = np.exp(-NN_ho.model(train_homo_, training=False).numpy().reshape((len(train_homo_),)))
    pred_ho_3 = np.exp(-NN_ho.model(test_homo, training=False).numpy().reshape((len(test_homo),)))
    print('./trained_model2/trans/nat_dimer_homo_trans_%d'%len(train_homo_))

#     setting = {'activation':'tanh','nn_shape':(256,256,256),'batch_size':len(train_lumo_), 'training_steps':80000,\
#     'learning_rate': 0.0008*(1-i), 'decay_rate':0.95, 'decay_per_steps':1000, 'save_step':1000, 'drop_rate':0, 'save_path':'./trained_model/trans/nat_dimer_lumo_trans_%d'%len(train_lumo_),\
#     'seed':None, 'debug_traj':False, 'pre_trained_path':None}
#     NN_lu = nn.NN(setting_dict=setting)
#     NN_lu.train(train_lumo_,train_clumo_)
#     pred_lu_1 = np.exp(-NN_lu.model(lumo_pairs, training=False).numpy().reshape((len(lumo_pairs),)))
#     pred_lu_2 = np.exp(-NN_lu.model(train_lumo_, training=False).numpy().reshape((len(train_lumo_),)))
#     pred_lu_3 = np.exp(-NN_lu.model(test_lumo, training=False).numpy().reshape((len(test_lumo),)))
#     print('./trained_model2/trans/nat_dimer_lumo_trans_%d'%len(train_lumo_))
    
    error1 = np.mean(np.multiply(abs(pred_ho_1-c_homo), np.power(c_homo,-1))*100)
    error2 = np.mean(np.multiply(abs(pred_ho_2-(np.exp(-train_chomo_))), np.power(np.exp(-train_chomo_),-1))*100)
    error3 = np.mean(np.multiply(abs(pred_ho_3-(np.exp(-test_chomo))), np.power(np.exp(-test_chomo),-1))*100)
    error4 = np.mean(abs(pred_ho_2-np.exp(-train_chomo_))*1000 * 27.2114)
    error5 = np.mean(abs(pred_ho_3-np.exp(-test_chomo))*1000 * 27.2114)
    print('MAPE of full data set: %5.3f %% \nMAPE of training set with %d samples: %5.3f %% \nMAPE of testing set with %d samples: %5.3f %% '\
          %(error1,len(train_homo_),error2,len(test_homo),error3))
    print('\nMAE(meV) of training set with %d samples: %5.3f  \nMAE(meV) of testing set with %d samples: %5.3f \n'\
          %(len(train_homo_),error4,len(test_homo),error5))
    
#     error1 = np.mean(np.multiply(abs(pred_lu_1-c_lumo), np.power(c_lumo,-1))*100)
#     error2 = np.mean(np.multiply(abs(pred_lu_2-(np.exp(-train_clumo_))), np.power(np.exp(-train_clumo_),-1))*100)
#     error3 = np.mean(np.multiply(abs(pred_lu_3-(np.exp(-test_clumo))), np.power(np.exp(-test_clumo),-1))*100)
#     error4 = np.mean(abs(pred_lu_2-np.exp(-train_clumo_))*1000 * 27.2114)
#     error5 = np.mean(abs(pred_lu_3-np.exp(-test_clumo))*1000 * 27.2114)
#     print('MAPE of full data set: %5.3f %% \nMAPE of training set with %d samples: %5.3f %% \nMAPE of testing set with %d samples: %5.3f %% '\
#           %(error1,len(train_lumo_),error2,len(test_lumo),error3))
#     print('\nMAE(meV) of training set with %d samples: %5.3f  \nMAE(meV) of testing set with %d samples: %5.3f \n'\
#           %(len(train_lumo_),error4,len(test_lumo),error5))
