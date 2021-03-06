import os
import sys
import numpy as np
import json
import cv2

import tensorflow as tf


def vid_label(root_path, json_file, mode, num_class=8):
    """
    Read "Json File" and Extract Dataset Infomation
    """
    l2i = {'ball': 0, 'swing': 1, 'strike': 2, 'hit': 3, 'foul': 4, 'in play': 5, 'bunt': 6, 'hit by pitch': 7} # labels to Index
    dataset = []
    with open(json_file, 'r') as f:
        data = json.load(f)

    for vid in data.keys():
        if data[vid]['subset'] != mode:
            continue
        if not os.path.exists(os.path.join(root_path, vid + ".mp4")):
            continue

        # Data_labeling
        multi_label = np.zeros((num_class,), dtype='int')
        duration = data[vid]['end'] - data[vid]['start']

        label = data[vid]['labels']
        if (len(label)) > 0:
            for labs in label:
                multi_label[l2i[labs]] = 1

        dataset.append((vid, multi_label, duration))

    return dataset


def get_Frames(root_path, vid, max_length):
    frames = []

    current_vid_path = os.path.join(root_path, vid)
    current_vid_path = current_vid_path + ".mp4"
    cap = cv2.VideoCapture(current_vid_path)

    while True():
        masking = np.empty(0)
        ret, frame = cap.read()

        if ret is False:
            break

        frame = frame[:, :, [0, 1, 2]] # Channel - RGB
        frame = pixel_centering(frame) # Pixe
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    h, w, c = frames[0].shape # [frame height / frame width / frame channel]

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
    def __init__(self, split_file, mode, root_path, max_length):
        super(CreateMLBYoutubeDataset, self).__init__()
        self.data = vid_label(root_path, split_file, mode)
        self.split_file = split_file
        self.root_path = root_path
        self.max_length = max_length

    def __getitem__(self, index):
        vid, label, duration = self.data[index]
        frames, masking = get_Frames(self.root_path, vid, self.max_length) # Convert Video2Frame

        return tf.convert_to_tensor(frames, dtype=tf.float32), label, masking, vid

    def __len__(self):
        return len(self.data)

