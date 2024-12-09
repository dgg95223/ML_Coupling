import numpy as np
import copy
from ML_Coupling.gpr_frame import GPR
import matplotlib.pyplot as plt
import subprocess
subprocess.run('export TF_INTRA_OP_PARALLELISM_THREADS=40', shell=True)

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
for ii,i in np.ndenumerate(c_homo_tr):
    if i<=0.00000000:
        ihzero.append(ii)
        c_homo_tr[ii] = 1e-9
print('Number of points to be deleted for homo:  ',len(ihzero))
homo_pairs_tr  = np.delete(homo_pairs_tr , ihzero, 0)
c_homo_tr  = np.delete(c_homo_tr , ihzero, 0)

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
train_homo_pairs = homo_pairs_tr[:]
train_lumo_pairs = lumo_pairs_tr[:]

train_c_homo = -np.log(c_homo_tr)[:]
train_c_lumo = -np.log(c_lumo_tr)[:]

MPEs = []
MAPEs = []

# start gpr training
n_ensemble = 6

for i in range(4, n_ensemble):
    # randomly pick data for training
    train_homo = copy.deepcopy(train_homo_pairs)
    train_chomo = copy.deepcopy(train_c_homo)

    index = np.random.choice(len(train_c_homo), size=int(len(train_c_homo)* 0.1 * i), replace=False)

    train_homo_ = np.delete(train_homo,index,0)
    train_chomo_ = np.delete(train_chomo,index,0)
    
    ###

    train_lumo = copy.deepcopy(train_lumo_pairs)
    train_clumo = copy.deepcopy(train_c_lumo)

    index = np.random.choice(len(train_c_lumo), size=int(len(train_c_lumo)* 0.1 * i), replace=False)

    train_lumo_ = np.delete(train_lumo,index,0)
    train_clumo_ = np.delete(train_clumo,index,0)
    # the rest of data for testing
    iall = np.arange(len(train_c_homo))
    idiff = np.setdiff1d(iall,index)
    test_homo = np.delete(copy.deepcopy(train_homo_pairs),idiff,0)
    test_chomo = np.delete(copy.deepcopy(train_chomo),idiff,0)

    gpr = GPR(setting_dict={'alpha':0.02, 'kernel':'RBF', 'optimizer':'fmin_l_bfgs_b', 'length_scale':0.1, 'length_scale_bounds':(1e-2, 1e1)})
    dshape = train_homo_.shape
    new_train = train_homo_.reshape((dshape[0],dshape[1]*dshape[2]*dshape[3]))
    gpr.train(new_train, train_chomo_)
    # testing
    pred1, std1 = gpr.predict(train_homo_pairs.reshape((len(train_homo_pairs),2*8*8)))
    pred2, std2 = gpr.predict(train_homo_.reshape((len(train_homo_),2*8*8)))
    pred3, std3 = gpr.predict(test_homo.reshape((len(test_homo),2*8*8)))
    # MPE
    error1 = np.mean(np.multiply(pred1-train_c_homo, np.power(train_c_homo,-1))*100)
    error2 = np.mean(np.multiply(pred2-train_chomo_, np.power(train_chomo_,-1))*100)
    error3 = np.mean(np.multiply(pred3-test_chomo, np.power(test_chomo,-1))*100)
    MPEs.append((error1, error2, error3))
    
    print('Ensemble %d:'%i)
    print('MPE of full data set: %5.3f %% \nError of training set with %d samples: %5.3f %% \nError of testing set with %d samples: %5.3f %% '%(error1,len(train_homo_),error2,len(test_homo),error3))
    # MAPE
    error1 = np.mean(np.multiply(abs(pred1-train_c_homo), np.power(train_c_homo,-1))*100)
    error2 = np.mean(np.multiply(abs(pred2-train_chomo_), np.power(train_chomo_,-1))*100)
    error3 = np.mean(np.multiply(abs(pred3-test_chomo), np.power(test_chomo,-1))*100)
    print('MAPE of full data set: %5.3f %% \nError of training set with %d samples: %5.3f %% \nError of testing set with %d samples: %5.3f %% '%(error1,len(train_homo_),error2,len(test_homo),error3))
    print('\n')
    MAPEs.append((error1, error2, error3))
    
    # plotting
    x1 = np.exp(-pred1) * 27.211
    x2 = np.exp(-pred2) * 27.211
    x3 = np.exp(-pred3) * 27.211
    y1 = np.exp(-train_c_homo) * 27.211
    y2 = np.exp(-train_chomo_) * 27.211
    y3 = np.exp(-test_chomo) * 27.211
    x0 = [0,1.25]
    y0 = [0,1.25]
    fig, ax = plt.subplots()
    ax.scatter(x2,y2, color='g')
    ax.scatter(x3,y3, color='b')
    ax.plot(x0,y0, color='r')
    ax.set_xlim(0,1.25)
    ax.set_ylim(0,1.25)
    ax.set_xlabel('GPR Electron Coupling (eV)')
    ax.set_ylabel('CDFT Electron Coupling (eV)')
    # ax.set_title('Error: %5.3f%%'%error)
    ax.set_aspect('equal')
    plt.savefig('./plot_results/gpr_homo_%d.png'%len(train_homo_))
