# Author: Jingheng Deng
'''
This script can generate massive qchem inputs for geometry scanning jobs with a initial xyz file and a template of qchem input file.
In this case, the relative translational distance between 2 molecules/fragments is scanned.
'''

import numpy as np
import subprocess

def read_xyz(filename, index=None, output='regular'):
    with open(filename,'r') as xyz:
        molecules = xyz.readlines()

    # clear unnecessary empty rows
    reverse_i = list(range(0, len(molecules)))[::-1]
    for i in reverse_i:
        if molecules[i] == '\n':
            if (len(molecules[i-1]) > 10) or (len(molecules[i-1]) == 1):
                molecules.pop(i)

    # get the number of atoms in each geometry
    atoms_num = []
    ii = 0
    while ii < len(molecules) :
        atoms_num.append(int(molecules[ii]))
        ii += (2 + int(molecules[ii]))
        if ii == len(molecules):
            break

    # get the amount of geometries
    geoms_num = len(atoms_num)
    atom_symbol = []
    # get the symbol of atoms in each geometry
    _atom_symbol = np.loadtxt(filename, usecols=0, dtype='str')
    start = 1
    for i in range(0, geoms_num):
        end = start + atoms_num[i]
        atom_symbol.append(_atom_symbol[start:end])
        start = end + 1

    if index is None:
        _index = -1  # read the last geometry as default
    elif index == 0: # read all geometries
        pass
    elif index > 0: # read the N^th geometry
        _index = index - 1
    elif index <= -1:
        _index = geoms_num + index

    # index == 'N' read the N^th geometry
    if output == 'regular':
        _geom = []
        for j in range(0, atoms_num[_index]):
            _geom_ = molecules[sum(np.add(atoms_num,2)[:_index]) + 2 + j].split()
            _geom.append(_geom_[1:4])
        _geom =np.array(_geom, dtype=np.float64)
    elif output == 'pyscf':
        _geom = ''
        for j in range(0, atoms_num[_index]):
            _col = molecules[sum(np.add(atoms_num,2)[:_index]) + 2 + j].split()[0:4]
            _geom_ = '%2s %12s %12s %12s\n'%(_col[0], _col[1], _col[2], _col[3])
            _geom += _geom_

    geoms = _geom
    atoms_num = atoms_num[_index]
    atom_symbol =atom_symbol[_index]

    return atoms_num, atom_symbol, geoms

xyz_path = '/home/jingheng/ML_data_set/QC_coupling/raw/nat_dimer/xyz/nat_dimer_para_x0_y0_z35.xyz'
n_atom, atom_sym, geo = read_xyz(xyz_path, output='regular')
n_atom_frag1 = 18
n_atom_frag2 = n_atom - n_atom_frag1
geo_frag1 = geo[0 : n_atom_frag1]
geo_frag2 = geo[n_atom_frag1 : n_atom]
geo_frag2[:,2] = np.add(geo_frag2[:,2], 3.5)

# shift frag1
geo_frag1_ = np.zeros(geo_frag1.shape)
geo_ = np.zeros(geo.shape)

# specify the scanning range
x_rot = np.arange(0.0, 91, 2)
y_rot = np.arange(0.0, 91, 2)
z_rot = np.arange(0.0, 91, 2)

qchem_input_template = '/home/jingheng/ML_data_set/QC_coupling/raw/nat_dimer/template_input.inp'
with open(qchem_input_template, 'r') as f:
    qc_input = f.readlines()

index = 0
for k in z_rot:
    k_ = k / 180 * np.pi
    z_rot_tm = np.array([[np.cos(k_), -np.sin(k_), 0],[np.sin(k_), np.cos(k_), 0], [0,0,1]])
    for i in x_rot:
        i_ = i / 180 * np.pi
        x_rot_tm = np.array([[1,0,0], [0, np.cos(i_), -np.sin(i_)],[0, np.sin(i_), np.cos(i_)]])
        input_path = 'zrot%03d_xrot%03d'%(int(k), int(i))
#       subprocess.run('mkdir '+ input_path, shell=True)
        for j in y_rot:
            j_ = j /180 * np.pi
            y_rot_tm = np.array([[np.cos(j_), 0, np.sin(j_)],[0,1,0], [-np.sin(j_), 0, np.cos(j_)]])
            geo_frag1_ = np.einsum('ij,jk,kl,lm->im', geo_frag1, x_rot_tm, y_rot_tm, z_rot_tm)

            index += 1
            geo_[0:n_atom_frag1]      = geo_frag1_
            geo_[n_atom_frag1:n_atom] = geo_frag2
            file_name = '/nat_dimer_para_xrot%03d_yrot%03d_zrot%03d.inp'%(int(i), int(j), int(k))
            print(input_path+file_name) 
                    
#           with open(input_path+file_name, 'w') as f:
#               f.write(''.join(qc_input[0:2]))
#               for l in range(0, n_atom):
#                   f.write('%s %15.8f %15.8f %15.8f\n'%(atom_sym[l], geo_[l][0], geo_[l][1], geo_[l][2]))
#               f.write(''.join(qc_input[2+n_atom:]))
#           with open(input_path+'/input_list', 'a+') as inp:
#               inp.write(file_name[1:-4]+'\n')                                    
