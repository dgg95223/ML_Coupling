import numpy as np
from cube2ovlp import load_cube, load_dat

def get_center(mo):
    '''return the indices of the center of the given matrix/tensor'''
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

def get_center_(mo_):
    '''return the indices of the center of a list of indices of a maatrix/tensor and the correspending values
    mo_ should have the following form [[(ix, iy, iz), v]]'''
    icoord = mo_[0]
    values = mo_[1]
    sum_qv = np.zeros(len(mo_[0]))
    sum_v = np.zeros(len(mo_[0]))
    qc = np.zeros(len(mo_[0]))

    for ii, i in np.ndenumerate(icoord):
        for j in range(0, len(icoord)):
            sum_qv[j] += i[j] * values[ii]
            sum_v[j] += i

    for k in range(0, len(sum_v)):
        qc[k] = sum_qv[k] / sum_v[k]

    return np.round(qc)
                
def get_cluster(imo): 
    '''mo_ is a 2-d array, [(ix,iy,iz), v], including the indices and the corresponding values
       nq is the size of mo tensor
    '''
    return get_dbscan_cluster(imo)

def get_dbscan_cluster(imo_):
    from sklearn.cluster import DBSCAN
    icluster = DBSCAN(eps=1, min_samples=5).fit_predict(imo_) #  eps=1, min_samples=5 have been tested 6/6/2022

    n_cluster = len(np.unique(icluster))
    if n_cluster == 1:
        print('There is probably something wrong with the clustering step, further check with visualization is recommanded.')
        # %matplotlib inline
        # %matplotlib widget
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(imo_[:, 0], imo_[:, 1], imo_[:, 2], c=icluster)
        plt.savefig('check_cluster.pdf')

    return n_cluster, icluster

def int_grids_cluster(cluster):
    for ii, i in np.ndenumerate(cluster):
        pass


def preprocess_mo(mo, thresh=1e-5):
    '''separate the mo to 2 parts, positive part and negative part'''
    positive = []
    negative = []
    for ii, i in np.ndenumerate(mo):
        if i > thresh:
            positive.append(ii)
        elif i < -thresh:
            negative.append(ii)

    return np.array(positive), np.array(negative)  # [(ix,iy,iz)] : n * 3 array

def make_mo_descriptor(cube_file): # dat file is also supported
    '''
    mo_plus, mo_minus:                   n_grids * n_dim array
    n_cluster:                           int
    icluster_plus, icluster_minus:       1 * n_grids array
    cluster_plus, cluster_minus:         n_cluster * 1 array
    '''
    # load mo form cube file and preprocess it to positive and negative part
    if cube_file.split('.')[-1] == 'cube':
        nq, dq, mo = load_cube(cube_file)
    elif cube_file.split('.')[-1] == 'dat':
        nq, dq, mo = load_dat(cube_file)
    mo_plus, mo_minus = preprocess_mo(mo) # 

    # clustering
    n_cluster_plus, icluster_plus = get_cluster(mo_plus)
    n_cluster_minus, icluster_minus = get_cluster(mo_minus)

    # match the clustered index to the values
    cluster_plus= []
    cluster_minus = []
    for i in range(0, n_cluster_plus):
        cluster = []
        for jj, j in enumerate(icluster_plus):
            if j == i:
                cluster.append(mo_plus[jj])
        cluster_plus.append(cluster)

    for i in range(0, n_cluster_minus):
        cluster = []
        for jj, j in enumerate(icluster_minus):
            if j == i:
                cluster.append(mo_minus[jj])
        cluster_minus.append(cluster)

    # get the center of each cluster
    center_plus = []
    center_minus = []

    for i in range(0, n_cluster_plus):
        center_plus.append(get_center_(cluster_plus[i]))
    for i in range(0, n_cluster_minus):
        center_minus.append(get_center_(cluster_minus[i]))
        
    # integrate value on all grids of each cluster
    int_plus = int_grids_cluster(cluster_plus)
    int_minus = int_grids_cluster(cluster_minus)

    return np.array((int_plus, int_minus)), np.array((center_plus, center_minus))