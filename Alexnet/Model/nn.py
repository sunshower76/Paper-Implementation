from abc import abstractmethod
import tensorflow as tf
import numpy as np
from Model.layers import conv_layer, max_pool, relu, fc_layer

class ConvNet(object):
    def __init__(self, **kwargs):
        self.input_shape = list(kwargs.get('input_shape'))
        self.num_classes = kwargs.get('num_classes')
        self.input = tf.compat.v1.placeholder(tf.float32, [None]+self.input_shape) # [None] + [257,257,3] = [None,257,257,3] , None = will be #batches
        self.label = tf.compat.v1.placeholder(tf.float32, [None]+[self.num_classes])
        self.is_train = tf.compat.v1.placeholder(tf.bool)

        self._build_model(**kwargs) # return dictionary
        self.loss = self._build_loss(**kwargs)


    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        Implement parts...
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        Build loss function
        Implement parts...
        """
        pass


    def predict(self, sess, dataset, verbose=False, **kwargs):
        return 0


class AlexNet(ConvNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_model(self, **kwargs):
        # parameter pop
        input_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = kwargs.get('num_classes')

        # The ratio of connected nodes, 1.0: no drop out,  0.5 : 50% connected(50% drop out)
        keep_prob = tf.cond(self.is_train,
                            lambda: 1. - dropout_prob,
                            lambda: 1.) # tf.cond(condiiton, true_function, false_function)
                                        # (lambda parameter:implementation)(input_to_para)

        # AlexNet input = input image - input RGB mean
        input = self.input - input_mean # (227, 227, 3)

        # dic['new key'] = value -> is the operation of add new 'key:value' pair
        with tf.compat.v1.variable_scope('conv1'):
            # (227, 227, 3) --> (55, 55, 96)
            conv_out_1 = conv_layer(input, 11, 4, 96, padding='VALID',
                                    weights_stddev=0.01, biases_value=0.0)
            #print('conv1.shape', conv_out_1.get_shape().as_list())

        relu_out_1 = relu(conv_out_1)
        maxpool_out_1 = max_pool(relu_out_1, 3, 2, padding='VALID') # (55, 55, 96) --> (27, 27, 96)
        #print('pool1.shape', maxpool_out_1.get_shape().as_list())

        with tf.compat.v1.variable_scope('conv2'):
            # (27, 27, 96) --> (27, 27, 256)
            conv_out_2 = conv_layer(maxpool_out_1, 5, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            #print('conv1.shape', conv_out_2.get_shape().as_list())

        relu_out_2 = relu(conv_out_2)
        maxpool_out_2 = max_pool(relu_out_2, 3, 2, padding='VALID') # (27, 27, 256) --> (13, 13, 256)
        #print('pool2.shape', maxpool_out_2.get_shape().as_list())

        with tf.compat.v1.variable_scope('conv3'):
            # (13, 13, 256) --> (13, 13, 384)
            conv_out_3 = conv_layer(maxpool_out_2, 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            #print('conv3.shape', conv_out_2.get_shape().as_list())

        relu_out_3 = relu(conv_out_3)
        # no max pool

        with tf.compat.v1.variable_scope('conv4'):
            # (13, 13, 384) --> (13, 13, 384)
            conv_out_4 = conv_layer(relu_out_3, 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            #print('conv4.shape', conv_out_4.get_shape().as_list())

        relu_out_4 = relu(conv_out_4)
        # no max pool

        with tf.compat.v1.variable_scope('conv5'):
            # (13, 13, 384) --> (13, 13, 256)
            conv_out_5 = conv_layer(relu_out_4, 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            #print('conv5.shape', conv_out_5.get_shape().as_list())

        relu_out_5 = relu(conv_out_5)
        maxpool_out_5 = max_pool(relu_out_5, 3, 2, padding='VALID') # (13, 13, 256) --> (6, 6, 256)
        #print('pool5.shape', maxpool_out_5.get_shape().as_list())

        # flatten feature map
        # (6, 6, 256) --> (,6x6x256)
        fm_dim = np.prod(maxpool_out_5.get_shape()[1:])
        f_emb = tf.reshape(maxpool_out_5, [-1, fm_dim])

        with tf.compat.v1.variable_scope('fc6'):
            # (,6x6x256) --> (,4096)
            fc_out_6 = fc_layer(f_emb, 4096,
                                weights_stddev=0.005, biases_value=0.1)

        relu_out_6 = relu(fc_out_6)
        drop_out_6 = tf.compat.v1.nn.dropout(relu_out_6, rate=1-keep_prob)
        #print('drop6.shape', drop_out_6.get_shape().as_list())

        with tf.compat.v1.variable_scope('fc7'):
            # (,4096 )--> (,4096)
            fc_out_7 = fc_layer(drop_out_6, 4096,
                                weights_stddev=0.005, biases_value=0.1)

        relu_out_7 = relu(fc_out_7)
        drop_out_7 = tf.compat.v1.nn.dropout(relu_out_7, rate=1-keep_prob)
        #print('drop7.shape', drop_out_7.get_shape().as_list())

        with tf.compat.v1.variable_scope('fc8'):
            # (,4096 ) --> (, num_classes)
            logits = fc_layer(drop_out_7, num_classes,
                              weights_stddev=0.01, biases_value=0.0)
        #print('logits.shape', logits.get_shape().as_list())

        # softmax
        pred = tf.nn.softmax(logits)

        self.logits = logits
        self.pred = pred

        return 0

    def _build_loss(self, **kwargs):
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.compat.v1.trainable_variables() # tf.get_varaible(trainable=True)로 선언된 모든 변수를 return (default : traible=True-> 사실상 모든 변수)
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables]) # 모든 가중치에 대해서 l2 loss 적용

        # Softmax cross-entropy loss function
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss

"""
 class VGGNet(ConvNet):

 class ResNet(ConvNet):
"""