import tensorflow as tf
import json

class NN(tf.keras.Model):
    def __init__(self, activation='tanh'):
        super().__init__()
        # initial load

        # inital NN
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu

        if nn_shape is None:
            nn_shape = (5)

        self.nn_shape = nn_shape
        self.batch_size = batch_size



        # initial train

        # initial test

    def build_data_set(self):
        file_name = self.file_name
        
        

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
        model = self.call()
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
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
