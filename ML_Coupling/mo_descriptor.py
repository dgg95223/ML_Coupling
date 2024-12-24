from ML_Coupling.cube2ovlp import load_cube, load_dat
import numpy as np

BOHR = 0.52917721092

class MO_descriptor():
    def __init__(self, cube_file):
        self.mo         = None
        self.imo_plus   = None
        self.imo_minus  = None
        self.cube_file  = cube_file

    def get_center(self, mo):
        '''return the index of the center of the given matrix/tensor'''
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
        '''return the indices of the center of a list of indices of a matrix/tensor'''
        icoord = imo
        values = mo

        sum_qv = np.zeros(len(icoord[0])) # 1 * 3 vecter
        sum_v  = np.zeros(len(icoord[0])) 
        qc     = np.zeros(len(icoord[0]))

        for i in icoord: 
            for j in range(0, len(icoord[0])):
                sum_qv[j] += i[j] * values[i]
                sum_v[j] += values[i]

        for k in range(0, len(sum_v)):
            qc[k] = sum_qv[k] / sum_v[k]

        return qc  # maybe should remove np.round() --10/18/2023

    def get_lobe(self, imo): 
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
    
    def ecu_dist2(self, coord1, coord2):
        ''' Sqare of Ecu distance'''
        dist = 0
        for i in range(0, len(coord1)):
            dist += np.power((coord1[i] - coord2[i]) * self.dq[i,i],2) 
            # print('ecu: ', dist)
            
        return dist        

    def int_grids_lobe(self, imo, mo):
        icoord = imo
        
        n_lobe = len(imo)
        int_values =  np.zeros(n_lobe)
        for ii, i in np.ndenumerate(icoord):
            for j in i:
                int_values[ii] += mo[j]
        
        return int_values
    
    def cal_r2_lobe(self, imo, mo, center):
        icoord = imo

        n_lobe = len(imo)
        r2_lobes = np.zeros(n_lobe)
        for ii, i in np.ndenumerate(icoord):
            for j in i:
                r2_lobes[ii] += mo[j] * abs(mo[j]) * self.ecu_dist2(j, center[ii[0]])
                
        return r2_lobes

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

    def make(self,c_type='int',unit='bohr'):
        '''
        mo:                                  nx * ny * nz, number of grids along each axis
        imo_plus, imo_minus:                 n_grids * n_dim array, indices of mo
        n_lobe:                              int, number of lobes
        ilobe_plus, ilobe_minus:             1 * n_grids array, indices of lobe
        lobe_plus, lobe_minus:               n_lobe * n_grids_lobe * n_dim, array, n_grids_lobe is not a fixed number, indices of values in mo tensor and values
        cneter_plus, center_minus:           n_lobe * n_dim, weighthed cneter of lobe, in a.u.
        '''
        # load mo form cube file and preprocess it to positive and negative part
        cube_file = self.cube_file
        if cube_file.split('.')[-1] == 'cube':
            self.nq, self.dq, self.mo = load_cube(cube_file)
            
        elif cube_file.split('.')[-1] == 'dat':
            self.nq, self.dq, self.mo = load_dat(cube_file)
        imo_plus, imo_minus = self.preprocess_mo(self.mo)  

        # clustering
        print('1 start clustering')
        n_lobe_plus, ilobe_plus = self.get_lobe(imo_plus)
        n_lobe_minus, ilobe_minus =self.get_lobe(imo_minus)

        lobe_plus= []
        lobe_minus = []
        for i in range(0, n_lobe_plus):
            lobe = []
            for jj, j in enumerate(ilobe_plus):
                if j == i:
                    lobe.append(imo_plus[jj])
            lobe_plus.append(lobe)

        for i in range(0, n_lobe_minus):
            lobe = []
            for jj, j in enumerate(ilobe_minus):
                if j == i:
                    lobe.append(imo_minus[jj])
            lobe_minus.append(lobe)

        # get the center of each cluster
        print('2 start getting center')
        center_plus = []
        center_minus = []

        for i in range(0, n_lobe_plus):
            center_plus.append(self.get_center_(lobe_plus[i], self.mo))
        for i in range(0, n_lobe_minus):
            center_minus.append(self.get_center_(lobe_minus[i], self.mo))
            
        if c_type == 'int':
            amp_plus = self.int_grids_lobe(lobe_plus, self.mo)
            amp_minus = self.int_grids_lobe(lobe_minus, self.mo)
        elif c_type == 'r2':
            amp_plus = self.cal_r2_lobe(lobe_plus, self.mo, center_plus)
            amp_minus = self.cal_r2_lobe(lobe_minus, self.mo, center_minus)
        # integrate values on all grids of each lobe
        

        amp, center = np.multiply((amp_plus, amp_minus),self.dq[0,0]**3), np.multiply((center_plus, center_minus), self.dq[0,0])
        mo_ = np.zeros((center.shape[1]*2, 4))
        amp = amp.flatten()
        center = center.reshape((center.shape[1]*2, 3))   # the unit of coordinates is a.u.

        if unit == 'bohr':
            pass
        elif unit =='ang':
            center = np.multiply(center,BOHR)

        for ii, i in enumerate(amp):
            mo_[ii] = np.append(amp[ii], center[ii])
        
        return mo_

class MO_pair_descriptor():
    def __init__(self, mo_des1, mo_des2):
        self.mo1 = mo_des1
        self.mo2 = mo_des2

    def make(self):
        mo1 = self.mo1
        mo2 = self.mo2
        # mo_pair = np.zeros((4, len(mo1), len(mo1))) # make \pho_i*\pho_j

        # mo_pair[0] = np.outer(mo1[:,0], mo2[:,0])
        # for i in range(0, 3):            # make q_i - q_j along x, y, z axis where q is generalized coordinate
        #     mo_pair[i+1] = np.log(np.outer(np.exp(mo1[:, i+1]), np.exp(-mo2[:, i+1])))
        mo_pair = np.zeros((2, len(mo1), len(mo1))) # make \pho_i*\pho_j
        dist = []

        mo_pair[0] = np.outer(mo1[:,0], mo2[:,0])
        for i in range(0, 3):            # make q_i - q_j along x, y, z axis where q is generalized coordinate
            dist_ = np.power(np.log(np.outer(np.exp(mo1[:, i+1]), np.exp(-mo2[:, i+1]))),2)
            dist.append(dist_)
        
        ecu_dist = np.sqrt(np.add(np.add(dist[0], dist[1]), dist[2]))
        mo_pair[1] = ecu_dist

        return mo_pair

