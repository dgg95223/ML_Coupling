import tensorflow as tf
import numpy as np
import json

class NN():
    def __init__(self, json_path):
        super().__init__()
        setting = json.load(json_path)
        self.setting = setting
        setting_ = {'activation':'tanh', 'nn_shape':(5,10), 'batch_size':16, 'epoch':1, 'learning_rate': 0.001} # default setting

        # inital NN
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
            self.lr = setting_['learning_rate']
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

        self.nbatch = self.ndata // self.batch_size

        # initial test

    def build_data_set(self, X, Y):
        data_set = tf.data.Dataset.from_tensor_slices((X,Y))
        return data_set

    def build_NN(self, data):
        self.flatten = tf.keras.layers.Reshape(target_shape=self.nn_shape)
        self.dense1 = tf.keras.layers.Dense(units=self.batch_size[0], activation=self.activation)
        self.dense2 = tf.keras.layers.Dense(units=self.batch_size[1])

        x = self.flatten(data)                     
        x = self.dense1(x)                      
        x = self.dense2(x)                      
        output = tf.nn.softmax(x)
        return output

    def train(self):
        self.model = self.build_NN()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=model)      # 实例化Checkpoint，设置保存对象为model
        for batch_index in range(1, num_batches+1):                 
            X, y = data_loader.get_batch(args.batch_size)
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
            if batch_index % 100 == 0:                              # 每隔100个Batch保存一次
                path = checkpoint.save('./save/model.ckpt')         # 保存模型参数到文件
                print("model saved to %s" % path)

    def test(self):
        self.model.evaluate(X,Y)


