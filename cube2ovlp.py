'''Convert standard Gaussian cube file of MO to ovlp of 2 MOs'''

import numpy as np
from pyrsistent import dq
from pyscf.data.nist import BOHR

def load_cube(cube_file):
    '''return the values on grids in a 3-d tensor array'''
    natom = np.loadtxt(cube_file, max_rows=1, skiprows=2, usecols=0, dtype=int)                        # number of atom  
    nq = np.loadtxt(cube_file, max_rows=3, skiprows=3, usecols=0, dtype=int)                           # number of grid point along x, y, z axis
    dq = np.loadtxt(cube_file, max_rows=3, skiprows=3, usecols=(1, 2, 3))                              # grid size vector along x, y, z axis, in Bohr
    value = np.loadtxt(cube_file, skiprows=6+natom, dtype=np.float64).reshape((nq[0],nq[1],nq[2]))     # values of mo1 on every grid point, in a.u.
    return nq, dq, value

def load_dat(dat_file):
    '''return the values on grids in a 3-d tensor array'''
    nq = np.loadtxt(dat_file, max_rows=3, skiprows=3, usecols=0, dtype=int)           # number of grid point along x, y, z axis
    dq = np.zeros((3,3))
    dq_ = np.loadtxt(dat_file, max_rows=3, skiprows=3, usecols=1)                     # grid size vector along x, y, z axis, in Bohr
    for ii, i in enumerate(dq_):
        dq[ii, ii] = i
    value = np.loadtxt(dat_file, skiprows=6, dtype=np.float64).reshape((nq[0],nq[1],nq[2]))                        # values of mo1 on every grid point, in a.u.
    return nq, dq, value

def rotate_grid(dq1, rot_the= 0, rot_phi=0, rot_gam=0):                        # not available yet 5/28/2022
    dq1 = dq1
    return  np.array([0,0,0], dtype=int)

def translate_grid(dq, trans_x=0, trans_y=0, trans_z=0):
    axis = ['x', 'y', 'z']
    grid_shifts = []
    for ii, i in enumerate([trans_x, trans_y, trans_z]):
        if i !=  0:
            print('Translation will occur along %s axis'%axis[ii])
            shift = i / BOHR                                                 # convert input translation from angstrom
            grid_shift = np.round(shift / dq[ii][ii])                        # convert xyz information to grid information
        else:
            print('No translation will occur along %s axis'%axis[ii])
            grid_shift = 0

        grid_shifts.append(grid_shift)

    return np.array(grid_shifts, dtype=int)                                             # 1d vector [x_shift, y_shift, z_shift] 

def cal_ovlp_mo(file1, file2, trans=None, rot=None):               # currently the part of rotation should not be right, make sure rot is set to None. 2022/5/29
    if trans is None:
        trans = np.zeros(3)
    else:
        pass
    if rot is None:
        rot = np.zeros(3)
    else:
        pass

    if file1.split('.')[-1] == 'cube':
        nq1, dq1, mo1 = load_cube(file1)
    elif file1.split('.')[-1] == 'dat':
        nq1, dq1, mo1 = load_dat(file1)

    if file2.split('.')[-1] == 'cube':
        nq2, dq2, mo2 = load_cube(file2)
    elif file1.split('.')[-1] == 'dat':
        nq2, dq2, mo2 = load_dat(file2)

    for i in (dq1-dq2).reshape(9):
        if i != 0.0:
            print('wrong')

    trans_shift = translate_grid(dq1, trans_x=trans[0], trans_y=trans[1], trans_z=trans[2])
    rot_shift   = rotate_grid(dq1, rot_the=rot[0], rot_phi=rot[1], rot_gam=rot[2])

    istart = np.array([0,0,0], dtype=int)
    iend   = np.array(nq1, dtype=int)

    # print('1',istart, iend, trans_shift)

    istart_ = np.add(istart, trans_shift)
    iend_   = np.subtract(iend, trans_shift)
    # print('2',istart_, iend_)

    ovlp = np.einsum('ijk,ijk->', mo1[istart[0]:iend_[0], istart[1]:iend_[1], istart[2]:iend_[2]], 
                                  mo2[istart_[0]:iend[0], istart_[1]:iend[1], istart_[2]:iend[2]]) * dq1[0,0] * dq1[1,1] * dq1[2,2]

    return ovlp

trans = [0,0,3.5]
cube_file1 = './overlap-1d/homo-xyz.dat'
cube_file2 = './overlap-1d/homo-xyz.dat'
ovlp=cal_ovlp_mo(cube_file1, cube_file2, trans)

print(ovlp)