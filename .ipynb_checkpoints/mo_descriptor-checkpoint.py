import numpy as np
from cube2ovlp import load_cube, load_dat

class MO_descriptor():
    def __init__(self, cube_file):
        self.mo         = None
        self.imo_plus   = None
        self.imo_minus  = None
        self.cube_file  = cube_file

    def get_center(self, mo):
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

    def get_center_(self, imo, mo):
        '''return the indices of the center of a list of indices of a maatrix/tensor'''
        icoord = imo
        values = mo

        sum_qv = np.zeros(len(icoord[0]))
        sum_v  = np.zeros(len(icoord[0]))
        qc     = np.zeros(len(icoord[0]))

        for i in icoord:
            for j in range(0, len(icoord[0])):
                sum_qv[j] += i[j] * values[i]
                sum_v[j] += values[i]

        for k in range(0, len(sum_v)):
            qc[k] = sum_qv[k] / sum_v[k]

        return np.round(qc)

    def get_cluster(self, imo): 
        return self.get_dbscan_cluster(imo)

    def get_dbscan_cluster(self, imo_):
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
        elif -1 in icluster:
            print('noise found during clustering process, clustering parameter should be modified.')

        return n_cluster, icluster

    def int_grids_cluster(self, imo, mo):
        icoord = imo
        
        n_cluster = len(imo)
        int_values =  np.zeros(n_cluster)
        for ii, i in np.ndenumerate(icoord):
            for j in i:
                int_values[ii] += mo[j]
        
        return int_values

    def preprocess_mo(self, mo, thresh=1e-5):
        '''separate the mo to 2 parts, positive part and negative part'''
        positive = []
        negative = []
        for ii, i in np.ndenumerate(mo):
            if i > thresh:
                positive.append(ii)
            elif i < -thresh:
                negative.append(ii)

        return positive, negative  # [(ix,iy,iz)] : n * 3 array

    def make(self):
        '''
        mo:                                  nx * ny * nz, number of grids along each axis
        imo_plus, imo_minus:                 n_grids * n_dim array, indices of mo
        n_cluster:                           int, number of clusters
        icluster_plus, icluster_minus:       1 * n_grids array, indices of cluster
        cluster_plus, cluster_minus:         n_cluster * n_grids_cluster * n_dim, array, n_grids_cluster is not a fixed number, indices of values in mo tensor and values
        cneter_plus, center_minus:           n_cluster * n_dim, weighthed cneter of cluster
        '''
        # load mo form cube file and preprocess it to positive and negative part
        cube_file = self.cube_file
        if cube_file.split('.')[-1] == 'cube':
            nq, dq, mo = load_cube(cube_file)
            
        elif cube_file.split('.')[-1] == 'dat':
            nq, dq, mo = load_dat(cube_file)
        imo_plus, imo_minus = self.preprocess_mo(mo)  
        self.mo = mo

        # clustering
        print('1 start clustering')
        n_cluster_plus, icluster_plus = self.get_cluster(imo_plus)
        n_cluster_minus, icluster_minus =self.get_cluster(imo_minus)

        cluster_plus= []
        cluster_minus = []
        for i in range(0, n_cluster_plus):
            cluster = []
            for jj, j in enumerate(icluster_plus):
                if j == i:
                    cluster.append(imo_plus[jj])
            cluster_plus.append(cluster)

        for i in range(0, n_cluster_minus):
            cluster = []
            for jj, j in enumerate(icluster_minus):
                if j == i:
                    cluster.append(imo_minus[jj])
            cluster_minus.append(cluster)

        # get the center of each cluster
        print('2 start getting center')
        center_plus = []
        center_minus = []

        for i in range(0, n_cluster_plus):
            center_plus.append(self.get_center_(cluster_plus[i], mo))
        for i in range(0, n_cluster_minus):
            center_minus.append(self.get_center_(cluster_minus[i], mo))
            
        # integrate values on all grids of each cluster
        int_plus = self.int_grids_cluster(cluster_plus, mo)
        int_minus = self.int_grids_cluster(cluster_minus, mo)

        int, center = np.multiply((int_plus, int_minus),dq[0,0]**3), np.multiply((center_plus, center_minus), dq[0,0])
        mo_ = np.zeros((center.shape[1]*2, 4))
        int = int.flatten()
        center = center.reshape((center.shape[1]*2, 3))

        for ii, i in enumerate(int):
            mo_[ii] = np.append(int[ii], center[ii])
        
        return mo_

class MO_pair_descriptor():
    def __init__(self, mo_des1, mo_des2):
        self.mo1 = mo_des1
        self.mo2 = mo_des2

    def make(self):
        mo1 = self.mo1
        mo2 = self.mo2
        mo_pair = np.zeros((4, len(mo1), len(mo1))) # make \pho_i*\pho_j

        mo_pair[0] = np.outer(mo1[:,0], mo2[:,0])
        for i in range(0, 3):            # make q_i - q_j along x, y, z axis where q is generalized coordinate
            mo_pair[i+1] = np.log(np.outer(np.exp(mo1[:, i+1]), np.exp(-mo2[:, i+1])))

        return mo_pair


