import tensorflow as tf
import numpy as np
import json

class NN():
    def __init__(self, json_path=None, setting_dict=None):
        setting_ = {'activation':'tanh', 'nn_shape':(240,240,240), 'batch_size':160, 'training_steps':100000,\
                    'learning_rate': 0.001, 'decay_rate':0.96, 'decay_per_steps':1000, 'drop_rate':0.5,\
                    'save_path':'./save/', 'save_step':10000 ,'seed':1, 'debug_traj':False}  # default setting   
        if json_path is not None:
            setting = json.load(json_path)
        elif (json_path is None) and (setting_dict is not None):
            setting = setting_dict
        else:
            print('No setting is specified, default setting will be applied.')
            setting = setting_
        self.setting = setting
        
        # inital NN
        if len(self.setting['nn_shape']) == 3:
            self.model = MLP(self.setting)
        else:
            self.model = MLP2(self.setting)

        if self.setting['activation'] == 'tanh':
            self.activation = tf.nn.tanh
        elif self.setting['activation'] == 'relu':
            self.activation = tf.nn.relu
        elif self.setting['activation'] == 'lrelu':
            self.activation = tf.nn.leaky_relu
        else:
            print('The chosen activation function is not available.')

        if setting['nn_shape'] is None:
            self.nn_shape = setting_['nn_shape']
        else:
            self.nn_shape = setting['nn_shape']

        if setting['drop_rate'] is None:
            self.drop_rate = setting_['drop_rate']
        else:
            self.drop_rate = setting['drop_rate']

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

        if setting['save_path'] is None:
            self.save_path = setting_['save_path']
        else:
            self.save_path = setting['save_path']
    
        # initial test
        self.accuracy = tf.keras.metrics.Accuracy()

        # initialize seed
        if setting['seed'] is None:
            self.seed = setting_['seed']
        else:
            self.seed = setting['seed']

        # tf.random.set_seed(self.seed) 

        if setting['debug_traj'] is None:
            self.debug_traj = setting_['debug_traj']
        else:
            self.debug_traj = setting['debug_traj']

    def build_data_set(self, X, Y):
        # load data from np.array
        data_set = tf.data.Dataset.from_tensor_slices((X,Y)).batch(self.batch_size).shuffle(self.batch_size)
        ndata = len(X)
        return data_set, ndata

    def loss_function(self):
        return tf.keras.losses.MeanSquaredError() # MSE loss function, only for test

    @tf.function
    def train_step(self, X, Y, is_save=None):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            # print('80:', Y.dtype, predictions.dtype)
            self.loss = self.loss_function()(Y, predictions)
            # self.loss = tf.losses.MAE(Y, predictions)
            # self.loss = tf.reduce_mean(self.loss)
        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if is_save is None:
            is_save = False
        
        if is_save is False:
            pass
        else:
            tf.print("loss: ", self.loss)

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
        # define optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        istep = 0
        while istep < self.training_steps:
            for X_, Y_ in self.train_data_set:
                if istep < self.training_steps:
                    # define learning rate
                    self.lr = tf.compat.v1.train.exponential_decay(self.lr_base, istep+1, self.decay_per_steps, self.decay_rate)
                    self.optimizer.learning_rate = self.lr
                    # one train step
                    if istep % self.save_step == 0:
                        is_save = True
                    else:
                        is_save = False
                    self.train_step(X_,Y_, is_save=is_save)
                    
                    # save model every selected steps
                    if istep % self.save_step == 0:
                        print('training step: %5d'%(istep))
                        checkpoint.save(self.save_path+'/ckpt/model_%06d.ckpt'%(istep))   # save model, not finished yet -- 2022/7/1
                        # print('training step: %5d, loss: %15.12f'%(istep, self.loss.numpy()))
                        
                        if (self.debug_traj is True) and istep >= 0:
                            import matplotlib as mpl
                            mpl.use('Agg')
                            import matplotlib.pyplot as plt

                            error = np.mean(np.multiply(self.model(X, training=False).numpy().reshape((len(Y),))-Y, np.power(Y,-1))*100)
                            x = np.linspace(0, 4, 41)
                            y = np.linspace(0, 4, 41)
                            Z = self.model(X, training=False).numpy().reshape((41,41))

                            fig, ax = plt.subplots()
                            ax.contourf(x,y, np.exp(-Z))
                            ax.set_title('Error: %f%%'%error)
                            plt.savefig('./traj/%d.jpg'%istep)
                            plt.close()
                else:
                    break
                istep += 1        
            self.train_data_set, self.ndata_train = self.build_data_set(X, Y) # one epoch finished so refresh the data set to start a new epoch

        tf.saved_model.save(self.model, self.save_path+'/model')

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
        # self.initializer = tf.keras.initializers.GlorotNormal(seed=self.setting['seed'])
        self.initializer = tf.keras.initializers.GlorotNormal()
        # self.input     = tf.keras.layers.InputLayer(input_shape=(4,8,8))
        self.input1    = tf.keras.layers.Flatten()
        self.input2    = tf.keras.layers.Flatten()
        self.input3    = tf.keras.layers.Flatten()
        self.input4    = tf.keras.layers.Flatten()
        self.concate   = tf.keras.layers.Concatenate()
        self.BN1       =tf.keras.layers.BatchNormalization()
        self.BN2       =tf.keras.layers.BatchNormalization()
        self.BN3       =tf.keras.layers.BatchNormalization()
        # self.dense0    = tf.keras.layers.Dense(4)
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.dropout1  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.dropout2  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        # self.dropout3  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.denseO    = tf.keras.layers.Dense(units=1)

    def call(self, inputs):

        # self.input_shape = inputs[0].numpy().shape
        x1 = self.input1(inputs[:,0,:])
        x2 = self.input2(inputs[:,1,:])
        x3 = self.input3(inputs[:,2,:])
        x4 = self.input4(inputs[:,3,:])
        # print(x2.shape)
        x = self.concate([x1,x2,x3,x4])
        x = self.dense1(x)  # hidden layer
        x = self.BN1(x)
        # x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        x = self.BN2(x)
        # x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        x = self.BN3(x)
        # x = self.dropout3(x)
        x = self.denseO(x)  # output layer
        output = x
        return output

class MLP_(tf.keras.Model):
    '''MO_pair_ (n,n) --> coupling(1,)'''
    def __init__(self, setting):
        super(MLP_, self).__init__()
        self.setting = setting
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.input1    = tf.keras.layers.Flatten()
        # self.dense0    = tf.keras.layers.Dense(4)
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer='zeros')
        # self.dropout1  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer='zeros')
        # self.dropout2  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer='zeros')
        # self.dropout3  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.denseO    = tf.keras.layers.Dense(units=1, kernel_initializer=self.initializer)

    def call(self, inputs):

        # self.input_shape = inputs[0].numpy().shape
        x = self.input1(inputs)
        # print(x2.shape)
        x = self.dense1(x)  # hidden layer
        # x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        # x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        # x = self.dropout3(x)
        x = self.denseO(x)  # output layer
        output = x
        return output

class MLP2(tf.keras.Model):
    '''MO_pair (4,n,n) --> coupling(1,)'''
    def __init__(self, setting):
        super(MLP2, self).__init__()
        self.setting = setting
        self.initializer = tf.keras.initializers.GlorotNormal(seed=self.setting['seed'])
        # self.input     = tf.keras.layers.InputLayer(input_shape=(4,8,8))
        self.input1    = tf.keras.layers.Flatten()
        self.input2    = tf.keras.layers.Flatten()
        self.input3    = tf.keras.layers.Flatten()
        self.input4    = tf.keras.layers.Flatten()
        self.concate   = tf.keras.layers.Concatenate()
        # self.dense0    = tf.keras.layers.Dense(4)
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer=self.initializer)
        # self.dropout1  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer=self.initializer)
        # self.dropout2  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer=self.initializer)
        # self.dropout3  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense4    = tf.keras.layers.Dense(units=self.setting['nn_shape'][3], activation=self.setting['activation'], kernel_initializer=self.initializer, bias_initializer=self.initializer)
        # self.dropout4  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.denseO    = tf.keras.layers.Dense(units=1, kernel_initializer=self.initializer)

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
        x = self.dense4(x)  # hidden layer
        # x = self.dropout4(x)
        x = self.denseO(x)  # output layer
        output = x
        return output

class MLP_Dexter(tf.keras.Model):
    '''2 * MO_pair 2 * (4,n,n) --> Dexter coupling(1,)'''
    def __init__(self, setting):
        super(MLP, self).__init__()
        self.setting = setting
        # self.initializer = tf.keras.initializers.GlorotNormal(seed=self.setting['seed'])
        self.initializer = tf.keras.initializers.GlorotNormal()
        # sub-net 1
        self.sub1_input1    = tf.keras.layers.Flatten()
        self.sub1_input2    = tf.keras.layers.Flatten()
        self.sub1_input3    = tf.keras.layers.Flatten()
        self.sub1_input4    = tf.keras.layers.Flatten()
        self.sub1_concate   = tf.keras.layers.Concatenate()
        self.sub1_BN1       =tf.keras.layers.BatchNormalization()
        self.sub1_BN2       =tf.keras.layers.BatchNormalization()
        self.sub1_BN3       =tf.keras.layers.BatchNormalization()
        # self.sub1_dense0    = tf.keras.layers.Dense(4)
        self.sub1_dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.sub1_dropout1  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.sub1_dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.sub1_dropout2  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.sub1_dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        # self.sub1_dropout3  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.sub1_denseO    = tf.keras.layers.Dense(units=1)

        # sub-net 2
        self.sub2_input1    = tf.keras.layers.Flatten()
        self.sub2_input2    = tf.keras.layers.Flatten()
        self.sub2_input3    = tf.keras.layers.Flatten()
        self.sub2_input4    = tf.keras.layers.Flatten()
        self.sub2_concate   = tf.keras.layers.Concatenate()
        self.sub2_BN1       =tf.keras.layers.BatchNormalization()
        self.sub2_BN2       =tf.keras.layers.BatchNormalization()
        self.sub2_BN3       =tf.keras.layers.BatchNormalization()
        # self.sub2_dense0    = tf.keras.layers.Dense(4)
        self.sub2_dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.sub2_dropout1  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.sub2_dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.sub2_dropout2  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.sub2_dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        # self.sub2_dropout3  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.sub2_denseO    = tf.keras.layers.Dense(units=1)

        # main net
        self.concate        = tf.keras.layers.Concatenate()
        self.dense1         = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
    def call(self, inputs):

        # self.input_shape = inputs[0].numpy().shape
        x1 = self.input1(inputs[:,0,:])
        x2 = self.input2(inputs[:,1,:])
        x3 = self.input3(inputs[:,2,:])
        x4 = self.input4(inputs[:,3,:])
        # print(x2.shape)
        x = self.concate([x1,x2,x3,x4])
        x = self.dense1(x)  # hidden layer
        x = self.BN1(x)
        # x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        x = self.BN2(x)
        # x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        x = self.BN3(x)
        # x = self.dropout3(x)
        x = self.denseO(x)  # output layer
        output = x
        return output

class _MLP(tf.keras.Model):  # baseline model
    '''MO_pair (4,n,n) --> coupling(1,)'''
    def __init__(self, setting):
        super(_MLP, self).__init__()
        self.setting = setting
        # self.initializer = tf.keras.initializers.GlorotNormal(seed=self.setting['seed'])
        self.initializer = tf.keras.initializers.GlorotNormal()
        # self.input     = tf.keras.layers.InputLayer(input_shape=(4,8,8))
        self.input1    = tf.keras.layers.Flatten()
        self.input2    = tf.keras.layers.Flatten()
        # self.input3    = tf.keras.layers.Flatten()
        # self.input4    = tf.keras.layers.Flatten()
        self.concate   = tf.keras.layers.Concatenate()
        self.BN1       =tf.keras.layers.BatchNormalization()
        self.BN2       =tf.keras.layers.BatchNormalization()
        self.BN3       =tf.keras.layers.BatchNormalization()
        # self.dense0    = tf.keras.layers.Dense(4)
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        # self.dropout1  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        # self.dropout2  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        # self.dropout3  = tf.keras.layers.Dropout(self.setting['drop_rate'])
        self.denseO    = tf.keras.layers.Dense(units=1)

    def call(self, inputs):

        # self.input_shape = inputs[0].numpy().shape
        x1 = self.input1(inputs[:,0])
        x2 = self.input2(inputs[:,1])
        
        # x3 = self.input3(inputs[:,2,:])
        # x4 = self.input4(inputs[:,3,:])
        # print(x2.shape)
        x = self.concate([x1,x2])
        x = self.dense1(x)  # hidden layer
        x = self.BN1(x)
        # x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        x = self.BN2(x)
        # x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        x = self.BN3(x)
        # x = self.dropout3(x)
        x = self.denseO(x)  # output layer
        output = x
        return output