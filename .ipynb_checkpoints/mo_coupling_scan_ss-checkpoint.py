import ML_Coupling.mo_descriptor as md
import ML_Coupling.nn_frame as nn
import numpy as np
import subprocess
import copy
subprocess.run('export TF_INTRA_OP_PARALLELISM_THREADS=52', shell=True)

'''
1. load mo_pair descriptor
'''

homo_pairs = np.load('./data/homo_homo_pair.npy')
lumo_pairs = np.load('./data/lumo_lumo_pair.npy')
homo_rot_pairs = np.load('./data/homo_pair_rot.npy')
lumo_rot_pairs = np.load('./data/lumo_pair_rot.npy')

'''
2. read coupling
'''
raw_data = np.loadtxt('./data/results.csv', delimiter=',',comments='#')
raw_data_rot = np.loadtxt('./data/results_rot.csv', delimiter=',',comments='#')
c_homo = abs(raw_data[:,3])
c_lumo = abs(raw_data[:,4])
c_homo_rot = abs(raw_data_rot[:,3])
c_lumo_rot = abs(raw_data_rot[:,4])

'''
3. collect data every n-1 points along each axis
'''
ss = [2, 4, 5]
for m in ss:
    step_size = m
    index = 0

    new_homo_pairs = []
    new_c_homo = []
    new_lumo_pairs = []
    new_c_lumo = []
    for ii, i in enumerate(c_homo):
        if index < len(c_homo):
            new_c_homo.append(c_homo[ii])
            new_homo_pairs.append(homo_pairs[ii])
            new_c_lumo.append(c_lumo[ii])
            new_lumo_pairs.append(lumo_pairs[ii])
        index += step_size

    new_homo_pairs = np.array(new_homo_pairs)
    new_c_homo = np.array(new_c_homo)
    new_lumo_pairs = np.array(new_lumo_pairs)
    new_c_lumo = np.array(new_c_lumo)
    print(len(new_homo_pairs))

    '''
    4. remove zero values for full data set
    '''
    ihzero = []
    for ii,i in np.ndenumerate(c_homo):
        if i<=0.00000000:
            ihzero.append(ii)
            c_homo[ii] = 1e-9
    print('Number of points to be deleted for homo:  ',len(ihzero))
    homo_pairs  = np.delete(homo_pairs , ihzero, 0)
    c_homo= np.delete(c_homo , ihzero, 0)

    ilzero = []
    for ii,i in np.ndenumerate(c_lumo):
        if i<=0.00000000:
            ilzero.append(ii)
            c_lumo[ii] = 1e-9
    print('Number of points to be deleted for lumo:  ',len(ilzero))
    lumo_pairs  = np.delete(lumo_pairs , ilzero, 0)
    c_lumo = np.delete(c_lumo , ilzero, 0)

    '''
    5. remove zero valuesfor selected data set
    '''
    ihzero = []
    for ii,i in np.ndenumerate(new_c_homo):
        if i<=0.00000000:
            ihzero.append(ii)
            new_c_homo[ii] = 1e-9
    print('Number of points to be deleted for homo:  ',len(ihzero))
    new_homo_pairs  = np.delete(new_homo_pairs , ihzero, 0)
    new_c_homo= np.delete(new_c_homo , ihzero, 0)

    ilzero = []
    for ii,i in np.ndenumerate(new_c_lumo):
        if i<=0.00000000:
            ilzero.append(ii)
            new_c_lumo[ii] = 1e-9
    print('Number of points to be deleted for lumo:  ',len(ilzero))
    new_lumo_pairs  = np.delete(new_lumo_pairs , ilzero, 0)
    new_c_lumo = np.delete(new_c_lumo , ilzero, 0)

    '''
    6. build training set
    '''
    train_homo_pairs = new_homo_pairs[:]
    train_lumo_pairs = new_lumo_pairs[:]

    train_c_homo = -np.log(new_c_homo)[:]
    train_c_lumo = -np.log(new_c_lumo)[:]

    train_homo = copy.deepcopy(train_homo_pairs)
    train_chomo = copy.deepcopy(train_c_homo)
    print('Size of full training set for homo:   ',len(train_chomo))
    index = np.random.choice(len(train_c_homo), size=int(len(train_c_homo)*0), replace=False)
    train_homo_ = np.delete(train_homo,index,0)
    train_chomo_ = np.delete(train_chomo,index,0)
    print('Size of selected training set for homo:   ',len(train_homo_))

    train_lumo = copy.deepcopy(train_lumo_pairs)
    train_clumo = copy.deepcopy(train_c_lumo)
    print('Size of full training set for lumo:   ',len(train_clumo))
    index = np.random.choice(len(train_c_lumo), size=int(len(train_c_lumo)*0), replace=False)
    train_lumo_ = np.delete(train_lumo,index,0)
    train_clumo_ = np.delete(train_clumo,index,0)
    print('Size of selected training set for lumo:   ',len(train_lumo_))

    '''
    7. build testing set
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
    8. load model and train, test
    ''' 
    setting = {'activation':'tanh','nn_shape':(256,256,256),'batch_size':len(train_homo_), 'training_steps':200000,\
    'learning_rate': 0.00008, 'decay_rate':0.95, 'decay_per_steps':1000, 'save_step':1000, 'drop_rate':0, 'save_path':'./trained_model/nat_dimer_homo_total_%d'%len(train_homo_),\
    'seed':None, 'debug_traj':False, 'pre_trained_path':'./trained_model/nat_dimer_33884/'}
    NN_ho = nn.NN(setting_dict=setting)
    NN_ho.train(train_homo_,train_chomo_)

    setting = {'activation':'tanh','nn_shape':(256,256,256),'batch_size':len(train_lumo_), 'training_steps':200000,\
    'learning_rate': 0.00008, 'decay_rate':0.95, 'decay_per_steps':1000, 'save_step':1000, 'drop_rate':0, 'save_path':'./nat_dimer_lumo_total_%d'%len(train_lumo_),\
    'seed':None, 'debug_traj':False, 'pre_trained_path':'./trained_model/nat_dimer_33884/'}
    NN_lu = nn.NN(setting_dict=setting)
    NN_lu.train(train_lumo_,train_clumo_)

    error1 = np.mean(np.multiply(abs(NN_ho.model(homo_pairs, training=False).numpy().reshape((len(homo_pairs),))+np.log(c_homo)), np.power(-np.log(c_homo),-1))*100)
    error2 = np.mean(np.multiply(abs(NN_ho.model(train_homo_, training=False).numpy().reshape((len(train_homo_),))-train_chomo_), np.power(train_chomo_,-1))*100)
    error3 = np.mean(np.multiply(abs(NN_ho.model(test_homo, training=False).numpy().reshape((len(test_homo),))-test_chomo), np.power(test_chomo,-1))*100)
    print('Error of full data set: %5.3f %% \nError of training set with %d samples: %5.3f %% \nError of testing set with %d samples: %5.3f %% '\
          %(error1,len(train_homo_),error2,len(test_homo),error3))

    error1 = np.mean(np.multiply(abs(NN_lu.model(train_lumo_pairs, training=False).numpy().reshape((len(train_lumo_pairs),))+np.log(c_lumo)), np.power(-np.log(c_lumo),-1))*100)
    error2 = np.mean(np.multiply(abs(NN_lu.model(train_lumo_, training=False).numpy().reshape((len(train_lumo_),))-train_clumo_), np.power(train_clumo_,-1))*100)
    error3 = np.mean(np.multiply(abs(NN_lu.model(test_lumo, training=False).numpy().reshape((len(test_lumo),))-test_clumo), np.power(test_clumo,-1))*100)
    print('Error of full data set: %5.3f %% \nError of training set with %d samples: %5.3f %% \nError of testing set with %d samples: %5.3f %% '\
          %(error1,len(train_lumo_),error2,len(test_lumo),error3))