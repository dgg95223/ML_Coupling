import numpy as np
from cube2ovlp import load_cube, load_dat

def get_center(mo):
    '''return the indices of the  center of the given matrix/tensor''' # need to modify 2022/6/2
    sum_qv = np.zeros(mo.ndim)
    sum_v = np.zeros(mo.ndim)
    qc = np.zeros(mo.ndim)
    for ii, i in np.ndenumerate(mo):
        for j in range(0, len(sum_qv)):
            sum_qv[j] += ii[j] * i
            sum_v[j] += i

    for k in range(0, len(sum_v)):
        qc[k] = sum_qv[k] / sum_v[k]

    return qc

def get_grid_pair(mo, nq):
    pairs = []
    grid_coord = [] 
    grids_neigh = []

    for ii, i in np.ndenumerate(mo):
        grid_coord.append(ii)

    for jj, j in enumerate(grid_coord):
        grid_this = j
        neighs = [j,j,j]
        for k in range(0, len(j)):
            if grid_this[k] < nq[k] - 1:
                neighs[k][k] = j[k] + 1
            elif grid_this[k] == nq[k] - 1:
                pass 
                

# def get_cluster(mo_, nq): 
#     '''mo_ is a 2-d array, [(ix,iy,iz), v], including the indices and the corresponding values
#        nq is the size of mo tensor
#     '''
#     x_next = np.array((1,0,0))
#     x_last = np.array((-1,0,0))
#     y_next = np.array((0,1,0))
#     y_last = np.array((0,-1,0))
#     z_next = np.array((0,0,1))
#     z_last = np.array((0,0,-1))

#     clusters = []


def get_dbscan_cluster(mo_):
    from sklearn.cluster import DBSCAN
    


def preprocess_mo(mo, thresh=1e-5):
    '''separate the mo to 2 parts, positive part and negative part'''
    positive = []
    negative = []
    for ii, i in np.ndenumerate(mo):
        if i > thresh:
            positive.append((ii,i))
        elif i < -thresh:
            negative.append((ii,i))

    return positive, negative  # [(ix,iy,iz), v]

def make_descrip_mo(file):
    if file.split('.')[-1] == 'cube':
        nq, dq, mo = load_cube(file)
    elif file.split('.')[-1] == 'dat':
        nq, dq, mo = load_dat(file)

    # scan every element to get the range of positive/negative block

    blocks = [] 

        