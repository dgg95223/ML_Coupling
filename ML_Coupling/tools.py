import numpy as np

def read_xyz(filename, index=None, output='regular'):
    '''
    index: '-1' refers to the last geometry
           'N' any integar larger than 0, refers to the N^th geometry, '-' refers to count the geometry in reversed order
           '0' refers to all geometry
    output mode: 'regular' output atom number, atom symbols, a np.array of coordinates
                 'pyscf' output atom number, atom symbols, a string includes atom symbols and coordinates  
    '''
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

    if index == 0:
        # read all geometries
        geoms = []
        for i in range(0, geoms_num):
            if output == 'regular':
                _geom = []
                for j in range(0, atoms_num[i]):
                    _geom_ = molecules[sum(np.add(atoms_num,2)[:i]) + 2 + j].split()[1:4]
                    _geom.append(_geom_)
                _geom =np.array(_geom, dtype=np.float64)
            elif output == 'pyscf':
                _geom = ''
                for j in range(0, atoms_num[i]):
                    _col = molecules[sum(np.add(atoms_num,2)[:i]) + 2 + j].split()[0:4]
                    _geom_ = '%2s %12s %12s %12s\n'%(_col[0], _col[1], _col[2], _col[3])
                    _geom += _geom_
                    # _geom = ''.join(molecules[sum(np.add(atoms_num,2)[:i]) + 2: sum(np.add(atoms_num,2)[:i]) + 2 + atoms_num[i]])
            geoms.append(_geom)
    else: 
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
            # _geom = ''.join(molecules[sum(np.add(atoms_num,2)[:_index]) + 2: sum(np.add(atoms_num,2)[:_index]) + 2 + atoms_num[_index]])
        geoms = _geom
        atoms_num = atoms_num[_index]
        atom_symbol =atom_symbol[_index]
    
    return atoms_num, atom_symbol, geoms

def check_close_mol(xyz1, xyz2):
    '''
    check if the two given molecule overlap/bonding with each other.
    '''
    bond_r ={'ch':2.9, 'cc':3.4, 'hh':2.4, 'co':3.22, 'oh':2.72, 'cn':3.25, 'nh':2.75}
    n_atom1, atom_sym1, mol1 = read_xyz(xyz1)
    n_atom2, atom_sym2, mol2 = read_xyz(xyz2)
    dist = []
    for i in range(0, 3):            # make q_i - q_j along x, y, z axis where q is generalized coordinate
        dist_ = np.power(np.log(np.outer(np.exp(mol1[:, i]), np.exp(-mol2[:, i]))),2)
        dist.append(dist_)
    ecu_dist = np.sqrt(np.add(np.add(dist[0], dist[1]), dist[2]))

    bonds = []
    for ii ,i in enumerate(atom_sym1):
        for jj, j in enumerate(atom_sym2):
            bonds.append(i+j)
            
    for ii,i in enumerate(bonds):
        if i.lower() == 'cc':
            bonds[ii] = 'cc'
        elif i.lower() == 'hh':
            bonds[ii] = 'hh'
        elif i.lower() == 'ch' or i.lower() =='hc':
            bonds[ii] = 'ch'
        elif i.lower() == 'co' or i.lower() == 'oc':
            bonds[ii] = 'co'
        elif i.lower() == 'oh' or i.lower() == 'ho':
            bonds[ii] = 'oh'
        elif i.lower() == 'cn' or i.lower() == 'nc':
            bonds[ii] = 'cn'
        elif i.lower() == 'nh' or i.lower() == 'hn':
            bonds[ii] = 'nh'

    bonds = np.array(bonds).reshape(ecu_dist.shape)
    overlap_bond = False

    for ii, i in np.ndenumerate(ecu_dist):
        bond_type = bonds[ii]
        if i < bond_r[bond_type]:
            overlap_bond = True
            print(ii,i,bond_type)
            break
        
    return overlap_bond

def cal_dexter_coupling(output, n_state=4, E_t=0.0):
    '''
    calculate dexter coupling from the Q-Chem output file
    the workiing equation is: V_DET = \sigma_{ij}(H_{Ii} * G_{ij} * H_{jF})
                            where G_{ij} = E_t * (I-H)^{-1}_{ij}
    H is the Hamiltonian matrix, ij are the indice for intermediate states,
    E_t is the tunneling energy whcih is zero for two identical molecules
    '''
    with open(output,'r') as o:
        out = o.readlines()
    
    for ii, i in enumerate(out):
        if ' CDFT-CI Hamiltonian matrix in orthogonalized basis\n' in i:
            H_str = ' '.join(out[ii+2 : ii+2+n_state]).split()
    H = []
    for i in H_str:
        if len(i) > 12:
            val_1 = i[0:-12]
            val_2 = i[-12:]
            if len(val_1) > 7:
                H.append(val_1)
            H.append(val_2)
        else:
            if len(i) > 7:
                H.append(i)

    H = np.array(H,dtype=np.float64).reshape((n_state,n_state))
    H_dia_min = np.min(H.diagonal())
    
    H_br = H[1:n_state-1,1:n_state-1]
    H_I_br = H[0,1:n_state-1]
    H_br_F = H[1:n_state-1,-1]
    
    I_br = np.identity(n_state-2)
    G_br = np.linalg.inv(E_t*I_br - (H_dia_min*I_br - H_br))

    V_dex = np.einsum('i,ij,j->', H_I_br,G_br,H_br_F)

    V_dex -= H[0,n_state-1]

    return V_dex
