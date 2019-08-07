import _pickle as cPickle
import urllib.request
import os
import tarfile
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from abc import *
from sklearn.preprocessing import OneHotEncoder

class DATA(object):
    def __init__(self, **kwargs):
        self.batch_start_index = 0
        self.epoch = 0
        self.resizing_size  = kwargs.get('input_image_size')
        self.batch_size = kwargs.get('batch_size')
        # None으로 된 것들은, child class에서 정의될 예정.
        self.set_size = None
        self.dummy_label = None
        self.imgpath_list = None
        self.labelpath_list = None

    #def one_hot(self):

    # resizing_size 필수 입력! (Network 입력 크기)
    @abstractmethod
    def load_data(self, subset_dir, one_hot=True, resizing_size=None):
        """
        implement how to load the data (ex, image and label path)
        You must save the list of the pathes of images  and pathes of labels
        """
        pass


    def next_batch(self, shuffle=True):
        print("present_batch : {}".format(self.batch_start_index))
        batch_size = self.batch_size
        # 이미지와 레이블을 담아둘 더미 행렬
        if self.epoch == 0 :
            self.X_set = np.empty((self.batch_size, self.resizing_size, self.resizing_size, 3), dtype=np.float32)
            self.dummy_label = list(range(self.set_size))
            np.random.shuffle(self.dummy_label)

        # 훈련한 이미지의 개수 + 배치 사이즈 > 존재하는 이미지 개수(즉, 한 epoch을 모두 돌기 바로 직전, 넘치는 경우)
        if (self.batch_start_index + batch_size > self.set_size):

            self.epoch += 1 # 1 epoch 증가
            print("epoch : {}".format(self.epoch))

            over_batch_size = self.batch_start_index + batch_size - self.set_size
            rest_batch_size = batch_size - over_batch_size

            batch_choice = self.dummy_label[self.batch_start_index : self.batch_start_index + rest_batch_size]
            batch_imgpath = self.imgpath_list[batch_choice]
            batch_labelpath = self.labelpath_list[batch_choice]

            for index, imgpath in enumerate(batch_imgpath):
                img = cv2.imread(imgpath)
                self.X_set[index] = img

            Y_set = self.Y_set[self.batch_start_index : self.batch_start_index+len(batch_labelpath)]

            # dummy label 초기화 후, 새로운 epoch을 수행하기 위한 준비
            self.batch_start_index = 0 # 초기화
            if (shuffle == True):
                np.random.shuffle(self.dummy_label)

            batch_choice = self.dummy_label[self.batch_start_index : self.batch_start_index + over_batch_size]
            batch_imgpath = self.imgpath_list[batch_choice]
            batch_labelpath = self.labelpath_list[batch_choice]

            for index, imgpath in enumerate(batch_imgpath):

                img = cv2.imread(imgpath)
                img = cv2.resize(img, dsize=(self.resizing_size, self.resizing_size), interpolation=cv2.INTER_AREA)
                self.X_set[index+rest_batch_size] = img

            Y_set = np.vstack(Y_set, self.Y_set[:len(batch_labelpath)] )

            self.batch_start_index += over_batch_size

        else:
            batch_choice = self.dummy_label[self.batch_start_index : self.batch_start_index+batch_size]
            batch_imgpath = self.imgpath_list[batch_choice]
            batch_labelpath = self.labelpath_list[batch_choice]

            for index, imgpath in enumerate(batch_imgpath):
                img = cv2.imread(imgpath)
                img = cv2.resize(img, dsize=(self.resizing_size, self.resizing_size), interpolation=cv2.INTER_AREA)
                self.X_set[index] = img

            Y_set = self.Y_set[self.batch_start_index : self.batch_start_index+len(batch_labelpath)]

            self.batch_start_index += batch_size

        return self.X_set, Y_set

class CIFAR10(DATA):
    # CIFAR-10 다운로드 및 압축 해제 함수
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 0

    # resizing_size 필수 입력! (Network 입력 크기)
    def load_data(self, subset_dir):
        img_label_name_list = os.listdir(subset_dir)

        imgname_list = np.array(img_label_name_list[0::2])  # mask indexing 하기 위해서.
        labelname_list = np.array(img_label_name_list[1::2])

        imgpath_list = np.core.defchararray.add(subset_dir + "\\", imgname_list)
        labelpath_list = np.core.defchararray.add(subset_dir + "\\", labelname_list)

        self.set_size = len(imgpath_list) # set크기, 나중에 step수 측정위해 사용됨.
        self.imgpath_list = imgpath_list
        self.labelpath_list = labelpath_list

        self.Y_set = np.empty((self.set_size), dtype=np.uint8)

        for index, labelpath in enumerate(labelpath_list):
            with open(labelpath, "r") as f:
                label = f.read()
                self.Y_set[index] = int(label)

        # one hot encoding
        # one hot encoding을 위해 sklearn에 맞게 형태변환 (2차원으로)
        self.Y_set = np.array(self.Y_set).reshape(-1, 1)  # one hot encoding 을 위한 형태로 변환 (2차원)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(self.Y_set)
        self.Y_set = enc.transform(self.Y_set).toarray()

        return 0



    def download(self, SAVING_PATH):
        DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        urllib.request.urlretrieve(DATA_URL, SAVING_PATH)  # 다운로드

        return 0

    def extract(self, ZIP_FILE_PATH, DEST_PATH):

        tarfile.open(ZIP_FILE_PATH, 'r:gz').extractall(DEST_PATH)  # 압축해제

        return 0

    def unpickle(self, PICKLE_PATH):
        with open(PICKLE_PATH, 'rb') as fo:
            dict = cPickle.load(fo, encoding='latin1')
        data = dict.get('data')
        labels = dict.get('labels')  # narray
        filenames = dict.get('filenames')
        size = len(data)

        return [size, data, labels, filenames]

    def pickle2image_label(self, return_unpickle, train_test_path, type):

        for i in range(return_unpickle[0]):
            self.step += 1

            flat_image = return_unpickle[1][i]
            label = return_unpickle[2][i]
            filename = return_unpickle[3][i]

            filename_split_dot = filename.split(".")
            class_name = filename_split_dot[0]
            img_type = filename_split_dot[1]

            filename_split_underbar = filename.split("_")
            number_and_type = filename_split_underbar[-1]
            img_number = number_and_type.split(".")[0]

            R = flat_image[0:1024].reshape((32, 32))
            G = flat_image[1024:2048].reshape((32, 32))
            B = flat_image[2048:3072].reshape((32, 32))
            img_array = np.dstack((R, G, B))

            if type == 0:
                cv2.imwrite(os.path.join(train_test_path, str(self.step).zfill(5)+"."+img_type), img_array)
                with open( os.path.join(train_test_path, str(self.step).zfill(5) + ".txt"), "w") as f:
                    f.write(str(label))
            else:
                cv2.imwrite(os.path.join(train_test_path, str(self.step).zfill(5)+"."+img_type), img_array)
                with open( os.path.join(train_test_path, str(self.step).zfill(5) + ".txt"), "w") as f:
                    f.write(str(label))

        return 0

    def decomp_tgz(self, unzip_folder):
        cifar10_batch_folder = os.path.join(unzip_folder, "cifar-10-batches-py")

        train_path = os.path.join(cifar10_batch_folder, "Train")
        test_path = os.path.join(cifar10_batch_folder, "Test")
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        #Train
        for i in range(1,6):
            train_batch = os.path.join( cifar10_batch_folder, "data_batch_" + str(i) )
            return_unpickle = self.unpickle(train_batch)
            self.pickle2image_label(return_unpickle, train_path, 0)

        # Test
        test_batch = os.path.join(cifar10_batch_folder, "test_batch")
        return_unpickle = self.unpickle(test_batch)
        self.pickle2image_label(return_unpickle, test_path, 1)

        return 0

#class AWA(object):

#class SUN(object):

#class CUB(object):



