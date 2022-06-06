import numpy as np
from cube2ovlp import load_cube, load_dat

def get_center(imo):
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
        print('There is probably something wrong the clustering step, further check with visualization isre commanded.')
        # %matplotlib inline
        # %matplotlib widget
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(imo_[:, 0], imo_[:, 1], imo_[:, 2], c=icluster)
        plt.savefig('check_cluster.pdf')

    return n_cluster, icluster

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

def make_mo_descriptor(cube_file):
    nq, dq, mo = load_cube(cube_file)
    mo_plus, mo_minus = preprocess_mo(mo)
    
    n_cluster_plus, icluster_plus = get_cluster(mo_plus)
    n_cluster_minus, icluster_minus = get_cluster(mo_minus)
    
    for i in [icluster_plus, icluster_minus]:
        for j in range(0, len(i)):
            