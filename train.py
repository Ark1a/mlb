import os
import numpy as np
import time

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
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

    :return 5+D Tensor [B, N, H, W, C], label, masking_layer
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

        :param temp_data_list:
        :return Frames [B, N, H, W, C]:
        """
        # Initialize
        batch_input = np.empty([self.batch_size, self.max_length, self.img_size[0], self.img_size[1], self.img_size[2]])
        batch_label = [None] * self.batch_size
        batch_masking = [None] * self.batch_size

        # Data generation
        for i, frames in enumerate(temp_data_list):
            # frames[0] = frames, frames[1] = labels, frames[2] = vid, frames[3] = masking
            batch_input[i, ] = frames[0]
            batch_label[i] = frames[1]
            batch_masking[i] = frames[3]

        batch_input = tf.convert_to_tensor(batch_input) # Convert ndarry to tensor
        batch_label = np.array(batch_label)
        batch_masking = np.array(batch_masking)
        batch_masking = tf.convert_to_tensor(batch_masking)

        return batch_input, batch_label, batch_masking

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
        x, y, masking = self.__data__generation(temp_data_list)

        return x, y, masking


def timer():
    current_time_info = time.strftime('%c', time.localtime(time.time()))
    return current_time_info


# Data pre-processing
train_dataset, test_dataset = CreateMLBYoutubeDataset(SPLIT_FILE_PATH, "training", ROOT_PATH, MAX_FRAME_LENGTH), CreateMLBYoutubeDataset(SPLIT_FILE_PATH, "testing", ROOT_PATH, MAX_FRAME_LENGTH)

# Define Dataloader
train_FE_dataloader, test_FE_dataloader = CNNDataLoader(train_dataset, BATCH_SIZE, MAX_FRAME_LENGTH, IMG_SIZE), CNNDataLoader(test_dataset, BATCH_SIZE, MAX_FRAME_LENGTH, IMG_SIZE)




# Define Functional API Model
# 수정필요!
model_input = tf.keras.Input(shape=(16, 224, 224, 3), name="video_frame")
masking_input = tf.keras.Input(shape=(16,), name="frame_masking")

x = tf.reshape(model_input, [-1, 224, 224, 3], name="5D_to_4D")
feature_extraction_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=True)
x = feature_extraction_model(x)
x = tf.reshape(x, [BATCH_SIZE, MAX_FRAME_LENGTH, 1000], name="Segment_feature")
# output = layers.GlobalMaxPooling2D()(x)

# 여기에 추가로 임베딩을 만든다?
# 임베딩을 한번에 다 박을수가 없으니까...Batch별로 embedding을 하는건가?

lstm = layers.LSTM(512)
"""
Masking vector를 어떤식으로 입력으로 줄 것인지

"""

x = lstm(x)

###

model = tf.keras.Model(model_input, x, name="feature_extraction")
model.summary()

predict_result = model.predict(test_FE_dataloader[0], verbose=1)

"""
LSTM의 인자에 직접적으로 Masking 정보를 대입할 수 있음.
masking 정보를 어떤식으로 줘야되지?
mask = [Batch, Time-sequence]

그럼 데이터로더를 통해서 마스크 정보를 따로 전달할 수 있는가?
- 이건 되는거 같은데
- masking 정보를 어떻게 하면 줄 수 있는지를 잘 모르겠네

"""
