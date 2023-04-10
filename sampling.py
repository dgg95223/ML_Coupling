import ML_Coupling.tools as tool
import numpy as np
import sys, time
from joblib import Parallel, delayed

# idx = int(sys.argv[1])
idx = 1
t0 = time.time()
xyz_path = './../ML_coupling/input/nat_dimer_para_x0_y0_z35.xyz'
n_atom, atom_sym, geo = tool.read_xyz(xyz_path, output='regular')
n_atom_frag1 = 18
n_atom_frag2 = n_atom - n_atom_frag1
geo_frag1 = geo[0 : n_atom_frag1]
geo_frag2 = geo[n_atom_frag1 : n_atom]

# specify the scanning range
x_shift = np.arange(0.0, 4.1, 0.1)
y_shift = np.arange(0.0, 0.1, 0.1)
z_shift = np.arange(5.8, 6.1, 0.1)
x_rot = np.arange(0.0, 91, 2)
y_rot = np.arange(0.0, 91, 2)
z_rot = np.arange(0.0, 91, 2)

# shift frag1
geo_frag1_ = np.zeros(geo_frag1.shape)
geo_ = np.zeros(geo.shape)
check = tool.check_close_mol

def scan_mix(i,j,k,i_,j_,k_):
    geo1=geo_frag1
    geo2=geo_frag2
    geo_ = np.zeros(geo1.shape)
    sym = atom_sym
    geo_[:,0] = np.add(geo1[:,0],i)
    geo_[:,1] = np.add(geo1[:,1],j)
    geo_[:,2] = np.add(geo1[:,2],k)
    geo_ = np.einsum('ij,jk,kl,lm->im', geo_,\
               np.array([[1,0,0], [0, np.cos(i_), -np.sin(i_)],[0, np.sin(i_), np.cos(i_)]]),\
               np.array([[np.cos(j_), 0, np.sin(j_)],[0,1,0], [-np.sin(j_), 0, np.cos(j_)]]),\
               np.array([[np.cos(k_), -np.sin(k_), 0],[np.sin(k_), np.cos(k_), 0], [0,0,1]]))
    close = check(mol1=geo_, mol2=geo2, atom_sym1=sym[0:18], atom_sym2=sym[18:36])
    if close is False:
        return geo_#,(i,j,k,i_,j_,k_)

trans_rot = Parallel(n_jobs=-1)(delayed(scan_mix)(i,j,k,i_,j_,k_) for k in z_shift for i in x_shift for j in y_shift for k_ in z_rot*np.pi/180 for i_ in x_rot*np.pi/180 for j_ in y_rot*np.pi/180)
trans_rot = np.array(trans_rot,dtype=object)
np.save('trans_rot_geo_%d.npy'%idx, trans_rot)

print(trans_rot.shape)
t1 = time.time()
print(t1-t0)