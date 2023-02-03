import numpy as np
import matplotlib.pyplot as plt
import sys

saved_model_path = sys.argv[1]
model_type = sys.argv[2]

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

homo_pairs_tr  = np.delete(homo_pairs_tr , ihzero, 0)
c_homo_tr  = np.delete(c_homo_tr , ihzero, 0)

ilzero = []
for ii,i in np.ndenumerate(c_lumo_tr):
    if i<=0.00000000:
        ilzero.append(ii)
        c_lumo_tr[ii] = 1e-9

lumo_pairs_tr  = np.delete(lumo_pairs_tr , ilzero, 0)
c_lumo_tr  = np.delete(c_lumo_tr , ilzero, 0)

'''
4. build training set
'''
train_homo_pairs = homo_pairs_tr[:]
train_lumo_pairs = lumo_pairs_tr[:]

train_c_homo = -np.log(c_homo_tr)[:]
train_c_lumo = -np.log(c_lumo_tr)[:]

'''
5. plotting
'''
if (model_type == 'NN') or (model_type =='nn'):
    import tensorflow as tf
    NN = tf.keras.models.load_model(saved_model_path, compile=False)
    pred = NN(train_homo_pairs, training=False).numpy().reshape((len(train_homo_pairs),))
    
    x = np.exp(-pred) * 27.211
    y1 = np.exp(-train_c_homo) * 27.211
    y2 = train_c_lumo
    x0 = [0,1.25]
    y0 = [0,1.25]

    fig, ax = plt.subplots()
    ax.scatter(x,y1)
    ax.plot(x0,y0, color='r')
    ax.set_xlim(0,1.25)
    ax.set_ylim(0,1.25)
    ax.set_xlabel('NN Electron Coupling (eV)')
    ax.set_ylabel('CDFT Electron Coupling (eV)')
    # ax.set_title('Error: %5.3f%%'%error)
    ax.set_aspect('equal')
    plt.savefig('NN_homo_%d.png'%len(train_homo_pairs))