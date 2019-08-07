import os
import tensorflow as tf
from Dataset import dataset
from Model.nn import AlexNet as ConvNet
from Learning.optimizers import MomentumOptimizer as Optimizer
from Learning.evaluators import AccuracyEvaluator as Evaluator
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" 1. Set training hyper-parameters """
hp_dic = dict()
#image_mean = train_set.images.mean(axis=(0, 1, 2))    # mean image
#np.save('C:\\Users\\rkske\\Desktop\\asirra-dogs-cats-classification-master\\train_results\\asirra_mean.npy', image_mean)    # save mean image
#hp_dic['image_mean'] = image_mean

# FIXME: Training hyper-parameters
hp_dic['model'] = 'AlexNet'
hp_dic['input_image_size'] = 227 #W,H
hp_dic['input_shape'] =  np.array([227,227,3])
hp_dic['num_classes'] = 10
hp_dic['batch_size'] = 256
hp_dic['num_epochs'] = 300

hp_dic['augment_train'] = True
hp_dic['augment_pred'] = True

hp_dic['init_learning_rate'] = 0.01
hp_dic['momentum'] = 0.9
hp_dic['learning_rate_patience'] = 30
hp_dic['learning_rate_decay'] = 0.1
hp_dic['eps'] = 1e-8

# FIXME: Regularization hyper-parameters
hp_dic['weight_decay'] = 0.0005
hp_dic['dropout_prob'] = 0.5

# FIXME: Evaluation hyper-parameters
hp_dic['score_threshold'] = 1e-4


""" 2. Load datasets """
root_dir = 'C:\\Users\\rkske\\Desktop\\Dataset\\CIFAR10'   # FIXME
trainval_dir = os.path.join(root_dir, 'Train')

# Load trainval set and split into train/val sets
train_set = dataset.CIFAR10(**hp_dic)
train_set.load_data(trainval_dir) # 해당 셋에대한 이미지의 어레이 리스트와, one-hot 리스트를 할당 받는다.



""" 3. Build graph, initialize a session and start training """
# Initialize
graph = tf.compat.v1.get_default_graph()
config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True # GPU 사용 시 설정

model = ConvNet(**hp_dic)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, **hp_dic)

sess = tf.compat.v1.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, save_dir='C:\\Users\\rkske\\Desktop\\선우\\공부\\Alexnet\\train_results\\model_trained_weights' , verbose=True, **hp_dic)
