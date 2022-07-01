import tensorflow as tf
import numpy as np
import json

class NN():
    def __init__(self, json_path):
        setting = json.load(json_path)
        self.setting = setting
        setting_ = {'activation':'tanh', 'nn_shape':(240,240,240), 'batch_size':32, 'training_steps':10000,\
                    'learning_rate': 0.001, 'decay_rate':0.9, 'decay_per_steps':1000}  # default setting

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

        if setting['batch_siize'] is None:
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


        self.train_tot_step = tf.Variable(0, trainable=False, dtype=tf.int32)  # not finished yet -- 2022/7/1
    
        # initial test

    def build_data_set(self, X, Y):
        # load data from np.array
        data_set = tf.data.Dataset.from_tensor_slices((X,Y)).batch(self.batch_size).shuffl(self.batch_size)
        self.ndata = len(X)
        self.data_set = data_set

    def loss_function(self, true, predict):
        loss = tf.keras.losses.MeanSquaredError(true, predict) # MSE loss function, only for test
        return loss

    def train(self):
        # initialize NN model
        model = MLP(self.setting)
        # initialize data set
        self.build_data_set(self.X, self.Y)
        if self.ndata > self.training_steps:
            print('The training steps are not suffcient enough to cover all in data set.')

        istep = 0
        while istep < self.training_steps:
            for X, Y in self.data_set:
                if istep < self.training_steps:
                    self.lr = tf.train.exponential_decay(self.lr_base, istep+1, self.decay_per_steps, self.decay_rate)
                    self.optimizer = self.models.optimizers.Adam(learning_rate=self.lr).minimize(self.loss)
                    self.train_step(X,Y)
                    if istep % 100 == 0:
                        path = checkpoint.save('./save/model_%06s.ckpt'%(istep))      # save model, not finished yet -- 2022/7/1
                else:
                    break
                istep =+ 1                    
            self.build_data_set(self.X, self.Y) # one epoch finished so refresh the data set to start a new epoch

        
        checkpoint = tf.train.Checkpoint(myAwesomeModel=model)      # save model, not finished yet -- 2022/7/1
        # print("model saved to %s" % path)

    @tf.function
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            self.loss = self.loss_function(Y, predictions)
        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def test(self, X, Y):
        self.model.evaluate(X,Y)

    def save_model(self):
        tf.saved_model(self.model, './model')

class MLP(tf.keras.Model):
    def __init__(self, setting):
        super(MLP, self).__init__()
        self.setting = setting
        self.dense0    = tf.keras.layers.Dense(4)
        self.dense1    = tf.keras.layers.Dense(units=self.setting['nn_shape'][0], activation=self.setting['activation'])
        self.dropout1  = tf.nn.dropout(0.5)
        self.dense2    = tf.keras.layers.Dense(units=self.setting['nn_shape'][1], activation=self.setting['activation'])
        self.dropout2  = tf.nn.dropout(0.5)
        self.dense3    = tf.keras.layers.Dense(units=self.setting['nn_shape'][2], activation=self.setting['activation'])
        self.dropout3  = tf.nn.dropout(0.5)
        self.dense4    = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense0(inputs)
        x = self.dense1(x)  # hidden layer
        x = self.dropout1(x)
        x = self.dense2(x)  # hidden layer
        x = self.dropout2(x)        
        x = self.dense3(x)  # hidden layer
        x = self.dropout3(x)
        x = self.dense4(x)  # output layer
        output = x
        return output

