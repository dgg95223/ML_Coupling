import tensorflow as tf

class NN(tf.keras.Model):
    def __init__(self, nn_shape=(5,5,10), batch_size=(1024,10), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu

        self.nn_shape = nn_shape
        self.batch_size = batch_size
        self.flatten = tf.keras.layers.Reshape(target_shape=self.nn_shape)
        self.dense1 = tf.keras.layers.Dense(units=self.batch_size[0], activation=self.activation)
        self.dense2 = tf.keras.layers.Dense(units=self.batch_size[1])

    def call(self, inputs):
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output