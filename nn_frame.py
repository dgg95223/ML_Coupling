import tensorflow as tf
import numpy as np
import json

class NN():
    def __init__(self, json_path):
        setting = json.load(json_path)
        self.setting = setting
        setting_ = {'activation':'tanh', 'nn_shape':(5,10), 'batch_size':16, 'epoch':1, 'learning_rate': 0.001, 'learning_rate_decay':0.9} # default setting

        # inital NN
        self.model = MLP()
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
        if setting['learning_rate'] is None: # need to be rewritten to add deacy rate to lr
            self.lr = tf.train.exponential_decay(setting_['learning_rate'])
        else:
            self.lr = setting['learning_rate']

        if setting['batch_siize'] is None:
            self.batch_size = setting_['batch_size']
        else:
            self.batch_size = setting['batch_size']

        if setting['epoch'] is None:
            self.epoch = setting_['epoch']
        else:
            self.epoch = setting['epoch']

        self.optimizer = tf.keras.optimizers.Adam() # inital adam optimizer
    
        # initial test

    def build_data_set(self, X, Y):
        # load data from np.array
        data_set = tf.data.Dataset.from_tensor_slices((X,Y)).batch(self.batch_size)
        self.ndata = len(X)

        return data_set

    def loss_function(self, true, predict):
        loss = tf.keras.losses.MeanSquaredError(true, predict) # MSE loss function, only for test
        return loss

    def train(self):
        model = self.models.optimizers.Adam(learning_rate=self.lr)
        nbatches = int(self.ndata // self.batch_size * self.epoch)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=model)      # 实例化Checkpoint，设置保存对象为model
        for batch_index in range(1, nbatches+1):                 
            data = self.build_data_set(X,Y)
            with tf.GradientTape() as tape:
                y_pred = model(X)
                self.loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                self.loss = tf.reduce_mean(self.loss)
                print("batch %d: loss %f" % (batch_index, self.loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
            if batch_index % 100 == 0:                              # 每隔100个Batch保存一次.
                path = checkpoint.save('./save/model.ckpt')         # 保存模型参数到文件
                print("model saved to %s" % path)

    @tf.function
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            self.loss = self.loss_object(Y, predictions)
        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def test(self, X, Y):
        self.model.evaluate(X,Y)

    def save_model(self):
        tf.saved_model(self.model, './model')

class MLP(tf.keras.Model):
    def __init__(self, **setting):
        super(MLP, self).__init__()
        self.setting = setting
        # self.flatten = tf.keras.layers.Reshape(target_shape=self.setting['nn_shape'])
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

