import os
import numpy as np
import time
import random
import sys
import datetime
import json


import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.utils import Sequence

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocessing import CreateMLBYoutubeDataset


# Interactive GPU memory Allocation
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Setting Seed for Reproduction
def setting_seed(seed:int=42):
    """
    Setting Seeds for Reproduction

    "42, the answer to life universe and Everything"
        - The Hitchhiker's Guide to the Galaxy
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


setting_seed()


# Data path
SPLIT_FILE_PATH = '/home/gon/Desktop/mlb-youtube-master/data/mlb-youtube-segmented_modi.json'
NEGATIVE_SPLIT_FILE_PATH = '/home/gon/Desktop/mlb-youtube-master/data/mlb-youtube-negative_modi_split_preprocess.json'

ROOT_PATH = '/home/gon/Desktop/sampled_video_resize'
NEGATIVE_ROOT_PATH = '/home/gon/Desktop/sampled_noact_resize'

SAVE_DIR = '/home/gon/Desktop/save_dir'
MODEL_CHECK_POINT = '/home/gon/Desktop/model_check'

# Create Save_dir
file_name = os.path.basename(sys.argv[0])
file_name = file_name.replace('.py', '')
TRIAL_TIME = "%d:%d_%d-%d-%d" % (datetime.datetime.now().hour, datetime.datetime.now().minute,
                                 datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().year)
data_saving = file_name + '_' + TRIAL_TIME
per_test_saving = os.path.join(SAVE_DIR, data_saving)

per_test_ckpt = per_test_saving + '/ckpt'
per_test_ckpt_file = per_test_ckpt + '/' + file_name

# Model Hyper-parameter
BATCH_SIZE = 2
EPOCHS = 5
MAX_FRAME_LENGTH = 16
LSTM_DIM = 512
IMG_SIZE = (224, 224, 3)
TRAINABLE = False
LEARNING_RATE = 0.0001 # Default=0.0001

# Model EarlyStopper
early_stopper = EarlyStopping(patience=10)

# Timer
star_time = int(time.time())

# Create Folder to Save result
try:
    if not (os.path.isdir(per_test_saving)):
        os.makedirs(os.path.join(per_test_saving))
except OSError:
    print('Error : Creating save directory.')

try:
    if not(os.path.isdir(per_test_ckpt)):
        os.makedirs(os.path.join(per_test_ckpt))
except OSError:
    print('Error : Creating ckpt directory.')


# Model CheckPoint
checkpointer = ModelCheckpoint(
    filepath=per_test_ckpt,
    verbose=1,
    save_best_only=True
)


class DataLoader(Sequence):
    """
    Push Data into CNN for Feature Extraction

    :return 5+D Tensor [B, N, H, W, C], label
    """
    def __init__(self, dataset, batch_size, max_length, image_size, shuffle=False):
        super(DataLoader, self).__init__()
        self.data_list = np.arange(len(dataset))
        self.indexes = np.arange(len(dataset) - batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.img_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # self.indexes = np.arange(len(self.data_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data__generation(self, temp_data_list):
        """
        Generate data containing batch_size samples

        :param temp_data_list:
        :return Frames [B, N, H, W, C], labels(one hot):
        """
        # Initialize
        batch_input = np.empty([self.batch_size, self.max_length, self.img_size[0], self.img_size[1], self.img_size[2]])
        batch_label = [None] * self.batch_size
        batch_mask = [None] * self.batch_size
        batch_vid = [None] * self.batch_size

        # Data generation
        for i, frames in enumerate(temp_data_list):
            # frames[0] = frames / frames[1] = labels / frames[2] = mask / frames[3] = vid
            batch_input[i, ] = frames[0]
            batch_label[i] = frames[1]
            batch_mask[i] = frames[2]
            batch_vid[i] = frames[3]

        batch_input = tf.convert_to_tensor(batch_input, dtype=tf.float32) # Convert ndarry to tensor
        batch_label = np.array(batch_label)
        batch_mask = np.array(batch_mask)
        batch_vid = np.array(batch_vid)

        return batch_input, batch_label, batch_mask, batch_vid

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size)) # 4665 > 4640, drop 25

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
        x, y, masking, vid = self.__data__generation(temp_data_list)

        return x, y, masking, vid


class ActivityRecognition(Model):
    def __init__(self, dataloader):
        super(ActivityRecognition, self).__init__()
        self.dataloader = dataloader

        self.max_pool = layers.MaxPooling2D(7)
        self.avg_pool = layers.AveragePooling2D(7)
        self.masking = layers.Masking()

        backbone_model = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
        self.cnn = Model(backbone_model.inputs, [backbone_model.output, backbone_model.layers[-3].output])
        self.lstm = layers.LSTM(LSTM_DIM, return_sequences=True, return_state=True) # return all hidden state, return cell state

        self.dense1 = layers.Dense(512, activation='relu', dtype='float32')
        self.dense2 = layers.Dense(256, activation='relu', dtype='float32')
        self.dense3 = layers.Dense(8, activation='softmax')

    def call(self, x, cnn_training=None):
        input, label, mask, vid = x
        self.cnn.trainable = cnn_training

        # CNN Feature Extraction
        input_reshape = tf.reshape(input, [BATCH_SIZE * MAX_FRAME_LENGTH, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]])  # [Batch * Frames, 224, 224, 3]
        vector, feature = self.cnn(input_reshape)  # [32, 1000], [32, 7, 7, 2048]
        pooled_feature = self.avg_pool(feature)  # [1, 1, 2048]

        # Feature Extraction Reshape
        # vector_reshaped = tf.reshape(vector, [BATCH_SIZE, MAX_FRAME_LENGTH, 1000])  # High-level Feature, [1000]
        feature_reshaped = tf.reshape(pooled_feature, [BATCH_SIZE, MAX_FRAME_LENGTH, 2048])  # Pooled Feature, [1,1,2048]

        # Data Masking
        mask = tf.cast(mask, tf.float32)  # [True : 1, False : 0]
        # vector_masking = self.masking(vector_reshaped * mask[:, :, tf.newaxis])
        feature_masking = self.masking(feature_reshaped * mask[:, :, tf.newaxis])

        # LSTM Motion Information Extraction
        # v_hidden_state, v_last_state, v_last_cell = self.lstm(vector_masking, training=training)
        f_hidden_state, f_last_state, f_last_cell = self.lstm(feature_masking, training=training)

        # Activity Classification Part
        result = self.dense1(f_last_state)
        result = self.dense2(result)
        result = self.dense3(result)

        return result


class Attention(tf.keras.layers.Layer):
    def __init__(self, units, attention_dim):
        super(Attention, self).__init__()
        self.units = units
        self.attention_dim = attention_dim

       """
	Working on...
       """

def train(dataloader, num_epochs=EPOCHS):
    ar_model = ActivityRecognition(dataloader)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    tr_s_timer = time_to_string() # Timer

    preds = []
    labels = []

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(num_epochs):

        print('\n\nEpoch {} / {}'.format(epoch + 1, num_epochs))
        print('-' * 75)

        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()

        i = BATCH_SIZE
        for x in dataloader:
            _, label, _, _ = x

            with tf.GradientTape() as tape:
                pred = ar_model(x, False)
                loss_value = loss_object(y_true=label, y_pred=pred)

                # 21-04-23/todo : Multi-binear Classifier
                """
                모델파트 또는 여기서 이진 분류기의 구성이 필요
                """
                variable = ar_model.trainable_variables
                gradients = tape.gradient(loss_value, variable)

                optimizer.apply_gradients(zip(gradients, variable))

                epoch_loss_avg(loss_value)
                epoch_accuracy(label, pred)

            print("{0:0d} - {1:04d} / {2:0d}. Current Loss : {3:0.6f} Current Acc : {4:0.6f}".format(epoch, i, len(dataloader) * 2,
                                                                                                     epoch_loss_avg.result().numpy(), epoch_accuracy.result().numpy()))
            i += BATCH_SIZE

    end_time = int(time.time() - star_time) / 3600
    tr_e_timer = time_to_string()

    print("\nModel training Started in : [%s]" % tr_s_timer)
    print("Model training ended in   : [%s]" % tr_e_timer)

    # training_result
    train_loss_result, train_accuracy_result = epoch_loss_avg.result().numpy(), epoch_accuracy.result().numpy()

    # calc mean average Precision
    # preds = np.array(preds).reshape(-1, 2)
    # labels = np.array(labels).reshape(-1, 2)

    # mean_AP, per_class_precision = meanAveragePrecision(preds, labels)

    # Model Weight Saving
    ar_model.save_weights('%s' % per_test_ckpt_file)

    return train_loss_result, train_accuracy_result, end_time #, mean_AP, per_class_precision


def validation(dataloader):
    val_ar_model = ActivityRecognition(dataloader)
    val_ar_model.load_weights(per_test_ckpt_file)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    val_str_time = time.time()
    val_s_timer = time_to_string()  # Timer

    val_preds = []
    val_labels = []

    val_loss = tf.keras.metrics.Mean()
    val_acc = tf.keras.metrics.CategoricalAccuracy()

    i = BATCH_SIZE
    for x in dataloader:
        _, label, _, _ = x

        pred = val_ar_model(x, False)
        loss_value = loss_object(y_true=label, y_pred=pred)

        # todo : Multi-binear Classifier pt2
        # temp_pred = tf.argmax(pred)
        # val_preds.append(temp_pred)
        # val_labels.append(label)

        val_loss(loss_value)
        val_acc(label, pred)

        print("{0:04d} / {1:0d}. Current Loss : {2:0.6f} Current Acc : {3:0.6f}".format(i, len(dataloader) * 2,
                                                                                        val_loss.result().numpy(), val_acc.result().numpy()))
        i += BATCH_SIZE

    val_end_time = int(time.time() - val_str_time) / 3600
    val_e_timer = time_to_string()

    print("\nModel validation Started in : [%s]" % val_s_timer)
    print("Model validation ended in   : [%s]" % val_e_timer)

    val_preds = np.array(val_preds).reshape(-1, 2)
    val_labels = np.array(val_labels).reshape(-1, 2)

    val_loss, val_acc = val_loss.result().numpy(), val_acc.result().numpy()
    # mean_ap, per_class_precision = meanAveragePrecision(val_preds, val_labels)

    return val_loss, val_acc, val_end_time #, mean_ap, per_class_precision


def meanAveragePrecision(preds, labels):
    """
    calculate Average Precision per-class

    :param pred:
    :param label:
    :return meanAvearagePrecision:
    """
    class_number = len(labels[0])
    class_precision = np.zeros(class_number)

    for i in range(len(class_precision)):
        current_preds, current_labels = preds[:, i], labels[:, i]

        tp = int(np.sum([1 for i in range(len(current_labels)) if current_labels[i] == 1 if current_preds[i] == current_labels[i]]))
        fp = int(np.sum([1 for i in range(len(current_labels)) if current_labels[i] == 0 if current_preds[i] != current_labels[i]]))

        if tp is 0:
            class_precision[i] = 0
            continue

        precision = tp / (tp + fp)
        class_precision[i] = precision

    mean_average_precision = np.sum(class_precision) / len(class_precision)

    return mean_average_precision, class_precision


def time_to_string():
    current_time_info = time.strftime('%c', time.localtime(time.time()))
    return current_time_info


# Data pre-processing
train_dataset = CreateMLBYoutubeDataset(SPLIT_FILE_PATH, "training", ROOT_PATH, MAX_FRAME_LENGTH)
test_dataset = CreateMLBYoutubeDataset(SPLIT_FILE_PATH, "testing", ROOT_PATH, MAX_FRAME_LENGTH)

# Define Dataloader
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, MAX_FRAME_LENGTH, IMG_SIZE)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, MAX_FRAME_LENGTH, IMG_SIZE)

# test
train_loss, train_acc, train_time, mean_ap, per_class_precision = train(train_dataloader)
test_loss, test_acc, test_time, test_mean_ap, test_per_class_precision = validation(test_dataloader)
print("\n\ntrain_loss : {0:0.6f}, train_acc : {1:0.6f}, Mean_AP : {2:0.6f}, train_time : {3:0.3f} H".format(train_loss, train_acc, mean_ap, train_time))
print("\n\nval_loss : {0:0.6f}, val_acc : {1:0.6f}, test_mean_ap : {2:0.6f}, test_time : {3:0.3f} H".format(test_loss, test_acc, test_mean_ap, test_time))


# Save result to json
train_loss, train_acc, mean_ap, train_time, per_class_precision = float(train_loss), float(train_acc), float(mean_ap), float(train_time), list(per_class_precision)
test_loss, test_acc, test_mean_ap, test_time, test_per_class_precision = float(test_loss), float(test_acc), float(test_mean_ap), float(test_time), list(test_per_class_precision)

dump_data = {}

dump_data['train'] = {"train_loss":train_loss, "train_acc":train_acc, "train_mAP":mean_ap, "train_time":train_time, "per_class_precisions":per_class_precision}
dump_data['test'] = {"test_loss":test_loss, "test_acc":test_acc, "test_mAP":test_mean_ap, "test_time":test_time, "test_per_class_precisions":test_per_class_precision}

json.dump(dump_data, open('/%s/%s_result.json' % (per_test_saving, data_saving), 'w'))
