import os
import numpy as np
import time

import tensorflow as tf

from keras.optimizers import SGD
from keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Masking

from tensorflow.keras.utils import Sequence
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocessing import CreateMLBYoutubeDataset

# Interactive GPU memory Allocation
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Data path
SPLIT_FILE_PATH = '/home/gon/Desktop/mlb-youtube-master/data/mlb-youtube-segmented.json'
NEGATIVE_SPLIT_FILE_PATH = '/home/gon/Desktop/mlb-youtube-master/data/mlb-youtube-negative.json'
SAVE_DIR = '/home/gon/Desktop/save_dir'
ROOT_PATH = '/home/gon/Desktop/sampled_video' # Sub-sampled Video Folder
MODEL_CHECK_POINT = '/home/gon/Desktop/model_check'

# Model Hyper-parameter
split_file = SPLIT_FILE_PATH
root_path = ROOT_PATH
BATCH_SIZE = 32
MAX_FRAME_LENGTH = 16
IMG_SIZE = (224, 224, 3)

# Model ChdeckPoints
checkpointer = ModelCheckpoint(
    filepath=os.path.join('dataset', 'checkpoints', 'Resnet50V2.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True
)

# Model EarlyStopper
early_stopper = EarlyStopping(patience=10)


class CNNDataLoader(Sequence):
    """
    Push Data into CNN for Feature Extraction

    :return 4+D Tensor [B * N * H * W * C]
    """
    def __init__(self, dataset, batch_size, max_length, image_size, shuffle=False):
        self.data_list = np.arange(len(dataset))
        self.indexex = np.arange(len(dataset) - batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.img_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data__generation(self, temp_data_list):
        """
        Generate data containing batch_size samples

        + Creating 4+D Tensor
         - Orignial Input : (N, H, W, C) + Batch
         - 4+D : (Batch * N, H, W, C)

        :param temp_data_list:
        :return Frames:
        """
        # Initialize
        batch_input = np.empty([self.batch_size, self.max_length, self.img_size[0], self.img_size[1], self.img_size[2]])
        batch_label = [None] * self.batch_size

        # Data generation
        for i, frames in enumerate(temp_data_list):
            batch_input[i, ] = frames[0][0]
            batch_label[i] = frames[1]

        batch_input = tf.convert_to_tensor(batch_input) # Convert ndarry to tensor
        batch_label = np.array(batch_label)

        return batch_input, batch_label

    def __len__(self):
        return int(np.floor(len(self.data_list) / self.batch_size)) # 4665 > 4640, drop 25

    def __getitem__(self, index):
        """
        Generate one Batch of Data

        :param index:
        :return:
        """
        # Generate Batch Indexes
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Find Batch list
        temp_data_list = [self.dataset[k] for k in indexes]

        # Generate Data
        x, y = self.__data__generation(temp_data_list)
        x = tf.reshape(x, (self.batch_size * self.max_length, self.img_size[0], self.img_size[1], self.img_size[2]))

        return x, y


def define_fe_model():
    """
    Define Feature Extraction Model

    :return: model
    """
    # Include Top
    # - True  = (Batch * F, 1000) // False = (Batch * F, 7, 7, 2048)
    model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=True)
    model.summary()

    return model


def feature_extraction(dataloader):
    """
    Extract Features from Videos. Using Pre-trained Models

    :param: dataloader
    :return: features (Per video - Per frame features)
    """
    sp = time.time()
    fe_sp = time.strftime('%c', time.localtime(time.time()))

    model = define_fe_model()
    features = model.predict(dataloader, verbose=1)

    ep = time.time() - sp
    fe_ep = time.strftime('%c', time.localtime(time.time()))

    print('Feature Extraction Started in : [', fe_sp, ']')
    print('Feature Extraction Ended in   : [', fe_ep, ']')
    print('Total Spent Time : %.2f min' % (ep/60))

    # reshape [74240, 1000] > [Length of Dataloader, 16, 1000]
    features = extracted_features.reshape([len(dataloader), MAX_FRAME_LENGTH, features.shape[1]])

    return features


def create_masking(dataset):
    """
    일일히 비교하는건 시간이 너무 오래걸림.
    padding 하는 과정에서 Masking Layer return하도록 변경


    masking dummy frames for LSTM

    :param dataset:
    :return boolean tensor:
    """
    # Time Checker
    sp = time.time()
    fe_sp = time.strftime('%c', time.localtime(time.time()))

    zero_tensor = tf.zeros([224, 224, 3])
    masking_layer = np.empty([len(dataset), MAX_FRAME_LENGTH], dtype=bool)

    for i in range(len(dataset)):
        for j in range(MAX_FRAME_LENGTH):
            current_frame = dataset[i][0][j]

            compare_result = tf.math.equal(current_frame, zero_tensor)
            compare_result = tf.reshape(compare_result, [224 * 224 * 3, ])
            convert_result, _, _ = tf.unique_with_counts(compare_result)

            if len(convert_result) is 1 and convert_result[0].numpy() == True:
                masking_layer[i][j] = True
            else:
                masking_layer[i][j] = False

    ep = time.time() - sp
    fe_ep = time.strftime('%c', time.localtime(time.time()))

    print('Layer Masking Started in : [', fe_sp, ']')
    print('Layer Masking Ended in   : [', fe_ep, ']')
    print('Total Spent Time : %.2f min' % (ep/60))

    return tf.convert_to_tensor(masking_layer)


# Data pre-processing
train_dataset, test_dataset = CreateMLBYoutubeDataset(split_file, "training", root_path, MAX_FRAME_LENGTH), CreateMLBYoutubeDataset(split_file, "testing", root_path, MAX_FRAME_LENGTH)

# Feature Extraction
train_FE_dataloader, test_FE_dataloader = CNNDataLoader(train_dataset, BATCH_SIZE, MAX_FRAME_LENGTH, IMG_SIZE), CNNDataLoader(test_dataset, BATCH_SIZE, MAX_FRAME_LENGTH, IMG_SIZE)
extracted_features = feature_extraction(train_FE_dataloader) # [74240, 1000] >> [145 * 32, 16, 1000] >> [4640, 16, 1000]

"""
중간 정리
    - extracted_features = [4640, 16, 1000]
"""


# Custom training model
