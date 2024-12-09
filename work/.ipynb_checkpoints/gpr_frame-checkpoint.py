import numpy as np
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class GPR():
    def __init__(self, json_path=None, setting_dict=None):
        setting_ = {'alpha':1, 'kernel':'RBF', 'optimizer':'fmin_l_bfgs_b', 'length_scale':1.0, 'length_scale_bounds':(1e-2, 1e2)}  # default setting   
        if json_path is not None:
            setting = json.load(json_path)
        elif (json_path is None) and (setting_dict is not None):
            setting = setting_dict
        else:
            print('No setting is specified, default setting will be applied.')
            setting = setting_
        self.setting = setting

        if self.setting['kernel'] == 'RBF':
            self.kernel = RBF(length_scale=self.setting['length_scale'], length_scale_bounds=self.setting['length_scale_bounds'])
        
        self.alpha = self.setting['alpha']

    def train(self, X, Y):
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.model.fit(X, Y)

    def predict(self, X):
        x_pred, std_pred = self.model.predict(X, return_std=True)
        return x_pred, std_pred