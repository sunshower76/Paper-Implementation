import os
import numpy as np
import tensorflow as tf
from Dataset import dataset
from Model.nn import AlexNet as ConvNet
from Learning.evaluators import AccuracyEvaluator as Evaluator


""" 1. Load dataset """
root_dir = 'C:\\Users\\rkske\\Desktop\\Dataset\\CIFAR10'   # FIXME
testval_dir = os.path.join(root_dir, 'Test')

# Load test set
# Load trainval set and split into train/val sets
test_set = dataset.CIFAR10()
test_set.load_data(testval_dir) # 해당 셋에대한 이미지의 어레이 리스트와, one-hot 리스트를 할당 받는다.


""" 2. Set test hyperparameters """
hp_d = dict()
#image_mean = np.load('C:\\Users\\rkske\Desktop\\asirra-dogs-cats-classification-master\\train_results\\asirra_mean.npy')  #np.load('/tmp/asirra_mean.npy')    # load mean image
#hp_d['image_mean'] = image_mean

# FIXME: Test hyperparameters
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Build graph, load weights, initialize a session and start test """
# Initialize
graph = tf.compat.v1.get_default_graph()
config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], num_classes=2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.compat.v1.Session(graph=graph, config=config)
saver.restore(sess, 'C:\\Users\\rkske\\Desktop\\asirra-dogs-cats-classification-master\\train_results\\model.ckpt')    # restore learned weights # FIXME
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
