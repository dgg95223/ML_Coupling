import numpy as np
import subprocess
import sys, copy
import ML_Coupling.mo_descriptor2 as md

'''
prepare data_set
1. make mo_pair descriptor
'''
BOHR = 0.52917721092

x_shift = np.arange(0, 0.01, 0.1)
y_shift = np.arange(0, 0.01, 0.1)
z_shift = np.arange(0, 0.01, 0.1)

# x_shift = np.arange(0, 4.1, 0.2)
# y_shift = np.arange(0, 4.1, 0.2)
# z_shift = np.arange(4.0, 4.01, 0.1)

# lumo = md.MO_descriptor('./dist_nap.plots/mo.35.cube').make(c_type='int',unit='ang')
homo = md.MO_descriptor('./dist_nap.plots/mo.34.cube').make(c_type='int',unit='ang')

# lumo = md.MO_descriptor('../../data/lumo-s0.cube').make(c_type='int',unit='ang')
# homo = md.MO_descriptor('../../data/homo-s0.cube').make(c_type='int',unit='ang')
# print(homo.shape,lumo.shape)

# homo = np.zeros((8,4))
# lumo = np.zeros((8,4))
# homo[0:len(_homo)] = _homo
# lumo[0:len(_lumo)] = _lumo


# for the original pair of one mo and itself
homo_pair = md.MO_pair_descriptor(homo, homo).make()
# lumo_pair = md.MO_pair_descriptor(lumo, lumo).make()

std_shape_h = homo_pair.shape
std_shape_l = lumo_pair.shape


homo_pairs = np.zeros((len(x_shift)*len(y_shift)*len(z_shift),) + std_shape_h)
lumo_pairs = np.zeros((len(x_shift)*len(y_shift)*len(z_shift),) + std_shape_l)

homo_ = np.zeros(homo.shape)
lumo_ = np.zeros(lumo.shape)

for kk, k in enumerate(z_shift):
    for ii, i in enumerate(x_shift):
        for jj, j in enumerate(y_shift):
            idx = kk * len(x_shift) * len(y_shift) + ii * len(y_shift) + jj
            homo_[:,0] = np.add(homo[:,0],0)
            homo_[:,1] = np.add(homo[:,1],i / BOHR)
            homo_[:,2] = np.add(homo[:,2],j / BOHR)
            homo_[:,3] = np.add(homo[:,3],k / BOHR)

            # rot_x = 10 / 180 * np.pi    # rotation angle along x axis
            # rot_tm = np.array([[1,0,0], [0, np.cos(rot_x), -np.sin(rot_x)],[0, np.sin(rot_x), np.cos(rot_x)]]) # rotate along x

            # homo_[:,1:] = np.einsum('ij,jk->ik', homo_[:,1:], rot_tm)

            homo_pair_ = md.MO_pair_descriptor(homo, homo_).make()
            homo_pairs[idx] = homo_pair_

            # lumo_[:,0] = np.add(lumo[:,0],0)
            # lumo_[:,1] = np.add(lumo[:,1],i / BOHR)
            # lumo_[:,2] = np.add(lumo[:,2],j / BOHR)
            # lumo_[:,3] = np.add(lumo[:,3],k / BOHR)

            print('delta des:\n',homo-homo_)
            print('delta des:\n',homo-homo_)

            

            # # lumo_[:,1:] = np.einsum('ij,jk->ik', lumo_[:,1:], rot_tm)

            # lumo_pair_ = md.MO_pair_descriptor(lumo, lumo_).make()
            # lumo_pairs[idx] = lumo_pair_


print(homo_pairs.shape)
print(lumo_pairs.shape)

# np.save('../../data/homo_homo_pair_dist_nap.npy', homo_pairs)
# np.save('../../data/lumo_lumo_pair_dist_nap.npy', lumo_pairs)
# np.save('../../data/homo_homo_pair_dist_nap_ori.npy', homo_pairs)
# np.save('../../data/lumo_lumo_pair_dist_nap_ori.npy', lumo_pairs)
# homo_pairs = np.load('../data/homo_homo_pair2.npy')