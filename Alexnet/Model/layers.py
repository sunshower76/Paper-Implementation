import tensorflow as tf

def weight_variable(shape, stddev=0.01):
    weights = tf.compat.v1.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights

def bias_variable(shape, value=1.0):
    biases = tf.compat.v1.get_variable('bias', shape, tf.float32,
                           tf.constant_initializer(value=value))
    return biases

def conv2d(input, weights, stride, padding='SAME'):
    return tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding=padding)

def max_pool(input, ksize, stride, padding='VALID'):
    return tf.nn.max_pool2d(input, ksize=[1, ksize, ksize, 1],
                   strides=[1,stride, stride, 1], padding=padding)

def relu(input):
    return tf.nn.relu(input)

def conv_layer(input, ksize, stride, out_depth, padding='SAME', **kwargs):
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(input.get_shape()[-1]) #image shape : (height, width, channel)

    filters = weight_variable([ksize, ksize, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)

    return conv2d(input, filters, stride, padding=padding) + biases

def fc_layer(x, out_dim, **kwargs):
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim],
                              stddev=weights_stddev)  # shape :( n , out_dim) , (이미지(batch)수 , feature 수)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases

