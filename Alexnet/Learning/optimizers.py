import os
import time
from abc import abstractmethod
import tensorflow as tf
from Learning.utils import plot_learning_curve




class Optimizer(object):
    """Base class for gradient-based optimization algorithms."""

    def __init__(self, convNet, train_set, evaluator,  **kwargs):
        """
        Optimizer initializer.
        :param model: ConvNet itself,      the model to be learned. (eliminated)
        :param train_set: DataSet, training set to be used.
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: DataSet, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra arguments containing training hyper-parameters.
            - batch_size: int, batch size for each iteration.
            - num_epochs: int, total number of epochs for training.
            - init_learning_rate: float, initial learning rate.
        """
        self.convNet = convNet
        self.train_set = train_set
        self.evaluator = evaluator

        # Training hyper-parameters
        # 가변인자에 해당 값들을 사용자가 설정한 값들을 받아와서 할당하는 코드
        self.batch_size = kwargs.pop('batch_size', 256)  # batch_size가 존재하지 않으면 default 값으로 256이 return 된다.
        self.num_epochs = kwargs.pop('num_epochs', 320)  # num_epochs가 존재하지 않으면 default 값으로 320이 return 된다.
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)  # init_learning_rate가 존재하지 않으면 default 값으로 0.01이 return 된다.

        self.learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32)    # Placeholder for current learning rate
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """Reset some variables."""
        self.curr_epoch = 1 # current_epoch = 1
        self.num_bad_epochs = 0    # number of bad epochs, where the model is updated without improvement.
        self.best_score = self.evaluator.worst_score    # initialize best score with the worst one
        self.curr_learning_rate = self.init_learning_rate    # current learning rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        tf.train.Optimizer.minimize Op for a gradient update.
        This should be implemented, and should not be called manually.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate (if needed) on every epoch, by its own schedule.
        This should be implemented, and should not be called manually.
        """
        pass

    def _step(self, sess, **kwargs):
        """
        Make a single gradient update and return its results.
        This should not be called manually.
        :param sess: tf.compat.v1.Session.
        :param kwargs: dict, extra arguments containing training hyper-parameters.
            - augment_train: bool, whether to perform augmentation for training.
        :return loss: float, loss value for the single iteration step.
                y_true: np.ndarray, true label from the training set.
                y_pred: np.ndarray, predicted label from the model.
        """
        augment_train = kwargs.pop('augment_train', True)

        # Sample a single batch
        X, y_true = self.train_set.next_batch()


        # Compute the loss and make update
        _, loss, y_pred = \
            sess.run([self.optimize, self.convNet.loss, self.convNet.pred],
                     feed_dict={self.convNet.input: X, self.convNet.label: y_true,
                                self.convNet.is_train: True,
                                self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model.
        :param sess: tf.compat.v1.Session.
        :param save_dir: str, the directory to save the learned weights of the model. save할 디렉토리 경로
        :param details: bool, whether to return detailed results.
        :param verbose: bool, whether to print details during training.
        :param kwargs: dict, extra arguments containing training hyperparameters.
        :return train_results: dict, containing detailed results of training.
        """
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())    # initialize all weights

        train_results = dict()    # dictionary to contain training(, evaluation) results and details
        train_size = self.train_set.set_size
        print('train_size', train_size)
        print('batch_size',self.batch_size)
        print('num_epochs',self.num_epochs)
        num_steps_per_epoch = train_size // self.batch_size # 반복 횟수(step) / 1 epoch
        num_steps = self.num_epochs * num_steps_per_epoch # 총 스텝 수 = number of epochs * steps per epoch

        if verbose:
            print()
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time() # 시간측정 시작

        # Start training loop (훈련)------------------------------------------------------------------------------
        for i in range(num_steps+1):
            # Perform a gradient update from a single minibatch
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs) # 한 스텝씩 진행 !!
            step_losses.append(step_loss)

            # Perform evaluation in the end of each epoch, 한 epoch 종료시!!!!
            if i % num_steps_per_epoch == 0:
                # Evaluate model with current mini-batch, from training set
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)


                # validation set없으면 그냥 현재 상황 설명
                if verbose:
                    # Print intermediate results
                    print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'.format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                    # Plot intermediate results
                    plot_learning_curve(-1, step_losses, step_scores, eval_scores=None, mode=self.evaluator.mode, img_dir=save_dir)
                curr_score = step_score

                # Keep track of the current best model,
                # by comparing current score and the best score
                # 현재 스코어가 더 좋으면 베스트 스코어 갱신하고, 중간 가중치 저장
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))    # save current weights
                    print('saved weights properly')
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        #훈련 종료-------------------------------------------------------------------------------------------
        if verbose:
            print()
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))
        print('Done.')

        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results


class MomentumOptimizer(Optimizer):
    """Gradient descent optimizer, with Momentum algorithm."""

    def _optimize_op(self, **kwargs):
        """
        tf.train.MomentumOptimizer.minimize Op for a gradient update.
        :param kwargs: dict, extra arguments for optimizer.
            - momentum: float, the momentum coefficient.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.compat.v1.trainable_variables()
        return tf.compat.v1.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
                .minimize(self.convNet.loss, var_list=update_vars)

    # bad_epochs를 이용하여 larning rate갱신한다.
    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate, when evaluation score plateaus.
        :param kwargs: dict, extra arguments for learning rate scheduling.
            - learning_rate_patience: int, number of epochs with no improvement
                                      after which learning rate will be reduced.
            - learning_rate_decay: float, factor by which the learning rate will be updated.
            - eps: float, if the difference between new and old learning rate is smaller than eps,
                   the update is ignored.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        #bad epochs : loss가 더 나아지지 않은 epochs 수
        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # Decay learning rate only when the difference is higher than epsilon.

            if self.curr_learning_rate - new_learning_rate > eps: # eps : 최소 learning rate 이 이하로는 내리지 않음.
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0
