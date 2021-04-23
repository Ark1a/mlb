import os
import random
import numpy as np
import json
import cv2

import tensorflow as tf


def vid_label(root_path, positive, negative, mode, num_class=8):
    """
    Read "Json File" and Extract Dataset Infomation
    """
    l2i = {''
           '': 0, 'swing': 1, 'strike': 2, 'hit': 3, 'foul': 4, 'in play': 5, 'bunt': 6, 'hit by pitch': 7} # labels to Index
    dataset = []
    with open(positive, 'r') as f:
        data = json.load(f)

    for vid in data.keys():
        # Original video data labeling
        if data[vid]['subset'] != mode:
            continue
        if not os.path.exists(os.path.join(root_path, vid + ".mp4")):
            continue

        # Data_labeling
        # multi_label = np.zeros((num_class,), dtype='int')
        duration = data[vid]['end'] - data[vid]['start']
        label = 1

        dataset.append((vid, label, duration))

    with open(negative, 'r') as nf:
        neg_data = json.load(nf)

    for vid in neg_data.keys():
        label = 0
        duration = 0

        dataset.append((vid, label, duration))

    random.shuffle(dataset)

    return dataset


def get_Frames(root_path, neg_path, vid, max_length):
    frames = []

    if len(vid) < 10:
        root_path = neg_path

    current_vid_path = os.path.join(root_path, vid)
    current_vid_path = current_vid_path + ".mp4"

    if not os.path.isfile(current_vid_path):
        return np.zeros([1, 1, 1, 1]), 0

    cap = cv2.VideoCapture(current_vid_path)

    while True:
        masking = np.empty(0)
        ret, frame = cap.read()

        if ret is False:
            break

        frame = frame[:, :, [0, 1, 2]] # Channel - RGB
        frame = pixel_centering(frame) # Pixel Centering
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) > 1:
        frames = np.asarray(frames, dtype=np.float32)
        h, w, c = frames[0].shape # [frame height / frame width / frame channel]

    else:
        # 원본 영상에 Frame은 존재하나, while문에서는 파악되지 않는 경우, 강제적으로 하나의 프레임을 포착하도록 도와줌
        cap = cv2.VideoCapture(current_vid_path)
        ret, frame = cap.read()

        if ret is False:
            return np.zeros([1, 224, 224, 3], dtype=np.float32), 0  # 프레임이 하나도 잡히지 않는 경우, Dataloader에서 처리하기 위해 더미 frame 하나 return

        frame = frame[:, :, [0, 1, 2]]  # Channel - RGB
        frame = pixel_centering(frame)  # Pixel Centering
        frames.append(frame)

        frames = np.asarray(frames, dtype=np.float32)
        h, w, c = frames[0].shape  # [frame height / frame width / frame channel]

        cap.release()
        cv2.destroyAllWindows()

    # Setting Maximum number of using Frames + Padding
    if len(frames) >= max_length:
        frames = frames[:max_length]
        masking = np.repeat(True, max_length)

    elif len(frames) < max_length:
        insufficient_frames = max_length - len(frames)
        padded_frames = np.zeros([insufficient_frames, h, w, c])

        true_masking = np.repeat(True, len(frames))
        false_masking = np.repeat(False, insufficient_frames)

        frames = np.concatenate([frames, padded_frames])

        masking = np.concatenate([true_masking, false_masking])

    return np.asarray(frames, dtype=np.float32), masking


def pixel_centering(pic):
    pic = (pic / 125) - 1 # (1 ~ -1) # Distribution
    # pic = (pic / 255) # (0 ~ 1) # Min-Max Normalization

    return pic


class CreateMLBYoutubeDataset:
    def __init__(self, split_file, negative_split_file, mode, root_path, negative_root_path, max_length):
        super(CreateMLBYoutubeDataset, self).__init__()
        self.data = vid_label(root_path, split_file, negative_split_file, mode)
        self.split_file = split_file
        self.negative_split_file = negative_split_file
        self.root_path = root_path
        self.negative_root_path = negative_root_path
        self.max_length = max_length

    def __getitem__(self, index):
        vid, label, duration = self.data[index]
        frames, masking = get_Frames(self.root_path, self.negative_root_path, vid, self.max_length) # Convert Video2Frame

        return tf.convert_to_tensor(frames, dtype=tf.float32), label, masking, vid

    def __len__(self):
        return len(self.data)
