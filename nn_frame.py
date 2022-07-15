import tensorflow as tf
import numpy as np
import json

class NN():
    def __init__(self, json_path=None, setting_dict=None):
        setting_ = {'activation':'tanh', 'nn_shape':(240,240,240), 'batch_size':160, 'training_steps':100000,\
                    'learning_rate': 0.001, 'decay_rate':0.96, 'decay_per_steps':1000, 'save_step':10000}  # default setting
        if json_path is not None:
            setting = json.load(json_path)
        elif (json_path is None) and (setting_dict is not None):
            setting = setting_dict
        else:
            print('No setting is specified, default setting will be applied.')
            setting = setting_
        self.setting = setting
        
        # inital NN
        self.model = MLP(self.setting)

        if self.setting['activation'] == 'tanh':
            self.activation = tf.nn.tanh
        elif self.setting['activation'] == 'relu':
            self.activation = tf.nn.relu
        else:
            print('The chosen activation function is not available.')

        if setting['nn_shape'] is None:
            self.nn_shape = setting_['nn_shape']
        else:
            self.nn_shape = setting['nn_shape']

        # initial train
        if setting['learning_rate'] is None:
            self.lr_base = setting_['learning_rate']
        else:
            self.lr_base = setting['learning_rate']

        if setting['batch_size'] is None:
            self.batch_size = setting_['batch_size']
        else:
            self.batch_size = setting['batch_size']

        if setting['training_steps'] is None:
            self.training_steps = setting_['training_steps']
        else:
            self.training_steps = setting['training_steps']

        if setting['decay_per_steps'] is None:
            self.decay_per_steps = setting_['decay_per_steps']
        else:
            self.decay_per_steps = setting['decay_per_steps']

        if setting['decay_rate'] is None:
            self.decay_rate = setting_['decay_rate']
        else:
            self.decay_rate = setting['decay_rate']

        if setting['save_step'] is None:
            self.save_step = setting_['save_step']
        else:
            self.save_step = setting['save_step']
    
        # initial test
        self.accuracy = tf.keras.metrics.Accuracy()

    def build_data_set(self, X, Y):
        # load data from np.array
        data_set = tf.data.Dataset.from_tensor_slices((X,Y)).batch(self.batch_size).shuffle(self.batch_size)
        ndata = len(X)
        return data_set, ndata

    def loss_function(self):
        return tf.keras.losses.MeanSquaredError() # MSE loss function, only for test

    # @tf.function
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            # print('80:', Y, predictions)
            # self.loss = tf.losses.MSE(Y, predictions)
            self.loss = tf.losses.MAE(Y, predictions)
            self.loss = tf.reduce_mean(self.loss)
        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    # @tf.function
    def test_step(self, X, Y):
        Y_pred = self.model(X, training=False)
        # print('88',Y.shape, Y_pred.shape)
        self.accuracy.update_state(Y, Y_pred)

    def train(self, X, Y):
        # initialize checkpoint instance
        checkpoint = tf.train.Checkpoint(saved_model=self.model)
        # initialize data set
        self.train_data_set, self.ndata_train = self.build_data_set(X, Y)
        if self.ndata_train > self.training_steps:
            print('The training steps are not sufficient enough to cover all points in data set.')

        istep = 0
        while istep < self.training_steps:
            for X, Y in self.train_data_set:
                if istep < self.training_steps:
                    # define learning rate
                    self.lr = tf.compat.v1.train.exponential_decay(self.lr_base, istep+1, self.decay_per_steps, self.decay_rate)
                    # define optimizer
                    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
                    # one train step
                    self.train_step(X,Y)
                    # save model every selected steps
                    if istep % self.save_step == 0:
                        checkpoint.save('./save/model_%06s.ckpt'%(istep))   # save model, not finished yet -- 2022/7/1
                        print('training step: %5d, loss: %12.9f'%(istep, self.loss))
                        # print('training step: %5d'%(istep))
                else:
                    break
                istep += 1        
            self.train_data_set, self.ndata_train = self.build_data_set(X, Y) # one epoch finished so refresh the data set to start a new epoch

        tf.saved_model.save(self.model, './save/model')

    def test(self, X, Y):
        # initialize test data set
        self.test_data_set, self.ndata_test = self.build_data_set(X, Y)
        # run a test step
        for X, Y in self.test_data_set:
            self.test_step(X, Y)
        accracy = self.accuracy.result()
        return accracy
        
class MLP(tf.keras.Model):
    '''MO_pair (4,n,n) --> coupling(1,)'''
    def __init__(self, setting):
        super(MLP, self).__init__()
        self.setting = setting
        # self.input     = tf.keras.layers.InputLayer(input_shape=(4,8,8))
        self.input1    = tf.keras.layers.Flatten()
        self.input2    = tf.keras.layers.Flatten()
        self.input3    = tf.keras.layers.Flatten()
        self.input4    = tf.keras.layers.Flatten()
        self.concate   = tf.keras.layers.Concatenate()
        # self.dense0    = tf.keras.layers.Dense(4)
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.dropout1  = tf.keras.layers.dropout(0.5)
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.dropout2  = tf.keras.layers.dropout(0.5)
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        # self.dropout3  = tf.keras.layers.dropout(0.5)
        self.dense4    = tf.keras.layers.Dense(1)

    def call(self, inputs):

        # self.input_shape = inputs[0].numpy().shape
        x1 = self.input1(inputs[:,0,:])
        x2 = self.input2(inputs[:,1,:])
        x3 = self.input3(inputs[:,2,:])
        x4 = self.input4(inputs[:,3,:])
        # print(x2.shape)
        x = self.concate([x1,x2,x3,x4])
        x = self.dense1(x)  # hidden layer
        # x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        # x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        # x = self.dropout3(x)
        x = self.dense4(x)  # output layer
        output = x
        return output

class MLP2(tf.keras.Model):
    '''MO (4, n) * 2 -> S (1,) -> coupling (1,)'''
    def __init__(self, setting):
        super(MLP, self).__init__()
        self.setting = setting
        # self.input     = tf.keras.layers.InputLayer(input_shape=(4,8,8))
        self.input1    = tf.keras.layers.Flatten()
        self.input2    = tf.keras.layers.Flatten()
        self.input3    = tf.keras.layers.Flatten()
        self.input4    = tf.keras.layers.Flatten()
        self.concate   = tf.keras.layers.Concatenate()
        # self.dense0    = tf.keras.layers.Dense(4)
        self.sub1_dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.dropout1  = tf.keras.layers.dropout(0.5)
        self.sub1_dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.dropout2  = tf.keras.layers.dropout(0.5)
        self.sub1_dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])

        self.sub2_dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.dropout1  = tf.keras.layers.dropout(0.5)
        self.sub2_dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.dropout2  = tf.keras.layers.dropout(0.5)
        self.sub2_dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.dropout1  = tf.keras.layers.dropout(0.5)
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.dropout2  = tf.keras.layers.dropout(0.5)
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        # self.dropout3  = tf.keras.layers.dropout(0.5)
        self.dense4    = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # self.input_shape = inputs[0].numpy().shape
        x1 = self.input1(inputs[:,0,:])
        x2 = self.input2(inputs[:,1,:])
        x3 = self.input3(inputs[:,2,:])
        x4 = self.input4(inputs[:,3,:])
        # print(x2.shape)
        x = self.concate([x1,x2,x3,x4])
        x = self.dense1(x)  # hidden layer
        # x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        # x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        # x = self.dropout3(x)
        x = self.dense4(x)  # output layer
        output = x
        return output

