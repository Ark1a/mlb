from __future__ import print_function

import os
import sys
import numbers
import numpy as np
import argparse
import copy
import time
import json
import cv2

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision.models as models

from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchsummary import summary # param=[channels, H, W]

sys.path.append('/home/gon/Desktop/some_toy_model')
from model_resnet50 import Residual_network


# Initialize
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
l2i = {'ball': 0, 'swing': 1, 'strike': 2, 'hit': 3, 'foul': 4, 'in play': 5, 'bunt': 6, 'hit by pitch': 7}
PIXEL_SIZE = 226 # 210201 - 왜 224가 아닌 226일까?
BATCH_SIZE = 1
DEVICE = torch.device("cuda:0")
NUM_CLASSES = 8 # Type of labels
FE_MODE = 1 # Weight freeze

# Data paths
SPLIT_FILE_PATH = '/home/gon/Desktop/mlb-youtube-master/data/mlb-youtube-segmented.json'
NEGATIVE_SPLIT_FILE_PATH = '/home/gon/Desktop/mlb-youtube-master/data/mlb-youtube-negative.json'
SAVE_DIR = '/home/gon/Desktop/save_dir'
ROOT_PATH = '/home/gon/Desktop/bb_extracted_frames' # Extracted Frame Folders


# PART : DATASET
def video_to_tensor(pic):
    """
    convert a numpy.ndarray(A) to tensor(B)

    numpy.ndarray (A) - (T * H * W * C)
    tensor / torch.floattensor (B) - (C * T * H * W) # 210201 - 단순 계산의 편리를 위해?
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_RGB_frames(image_dir, vid, start, num):
    frames = []
    image_dir = os.path.join(image_dir, vid + '.mp4')

    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid + '.mp4' + '_' + str(i) + '.jpg'))[:, :, [2, 1, 0]] # BGR 2 RGB
        w, h, c = img.shape
        if w < PIXEL_SIZE or h < PIXEL_SIZE:
            # center cropping
            d = PIXEL_SIZE - min(w, h)
            sc = 1 + d/(min(w, h))
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img/255.) * 2 - 1
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def make_Dataset(split_file, mode, root_path, num_class=8):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():

        if data[vid]['subset'] != mode:
            continue
        if not os.path.exists(os.path.join(root_path, vid + ".mp4")):
            continue

        num_frames = len(os.listdir(os.path.join(root_path, vid + ".mp4")))

        # Data Info
        multilabel = np.zeros((num_class,))
        duration = data[vid]['end'] - data[vid]['start']
        fps = num_frames / duration

        label = data[vid]['labels']
        if(len(label)) > 0:
            for labs in label:
                multilabel[l2i[labs]] = 1

        dataset.append((vid, multilabel, duration, num_frames))
        i += 1

    return dataset


# Creating Dataset
class CreateMLBYoutubeDataset(data_utils.Dataset):
    def __init__(self, split_file, mode, transform, root_path, save_dir):
        self.data = make_Dataset(split_file, mode, root_path)
        self.split_file = split_file
        self.transform = transform
        self.root_path = root_path
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Argument : (int) index

        :returns
            tuples (image, target value) where target is class_index of the target class
        """
        vid, label, duration, nf = self.data[index]

        # 210201 - npy 파일로 만들지 말고 쓸 일이 있을까
        if os.path.exists(os.path.join(self.save_dir, vid + '.npy')):
            return 0, 0, vid

        imgs = load_RGB_frames(self.root_path, vid, 0, nf) # 0 => start_number
        # imgs = self.transform(imgs) # modification-210202

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)


# PART : DATA-PRE PROCESS
class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        th, tw = self.size

        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+th, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size) # {0} = 0th variable


# Training_model
def train_model(model, criterion, optimizer, scheduler, num_epoches, dataloaders, device, dataset_size):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict) # 단순 객체복사, Shallowcopy, Deepcopy
    best_acc = 0.0

    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches - 1))
        print('-' * 10)

        # 모델 학습 Part
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            # for inputs, labels in dataloaders[phase]:
            for data in dataloaders[phase]:

                inputs, labels, name = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward part of transfer learning # 어쩌면 이 부분이 내가 알고 싶었던 부분이 아닐까 싶은데
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # update loss and acc
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_acc.double() / dataset_size[phase]

            print('{} Loss : {:.4f}, ACC : {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 나도 선배처럼 시작 날짜, 요일, 종료 시간, Spent_time 이렇게 나오게 작성해볼까?
    print('Best Val Acc : {:4f}'.format(best_acc))

    model.load_state_dict(best_model_weights) # 가장 좋았던 놈으로 Return
    return model


def main():
    # define model
    model = models.resnet50(pretrained=True).cuda()
    summary(model, (3, 224, 224), batch_size=32)

    # Parser Argument
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-split_file_path', type=str)
    # parser.add_argument('-root_path', type=str)
    # parser.add_argument('-save_dir', type=str)
    # parser.add_argument('-fe_mode' type=int, default=0, help='')
    # parser.add_argument('-gpu', type=int, default=0)
    # parser.add_argument('-seed', type=int, default=1)
    # args = parser.parse_args()

    # Environment Setting with Parser Arguments
    os.environ["CUDA_DEVICE_ORDER"] = str(0)
    torch.manual_seed(1)

    """
    # revision part 1 
    
    - 최적화하고 싶은 지역 (1,2)
    
    - 정확하게 어떤 역할을 하나?
        - 영상으로부터 추출된 Frame을 이용하여 텐서 생성(T*H*W*C)
        - 텐서 외에도 모델에 사용될 여러 변수 선언 // (텐서(C*T*H*W), 레이블, vid)
        - 생성된 텐서 구조에 멀티 레이블 수행 # 이건 유지해도 될거같고
        - transform, transpose
        
    - 교수님께 보고드릴 내용은 있어야하니.. 우선은 이건은 추후에 진행하는걸로
    """
    test_transform = transforms.Compose(CenterCrop(224))

    # Call Create_Dataset
    # 210201 - 데이터셋 형태는?, 데이터로더를 이용해 한번에 입력으로 주어지는 데이터의 형태는? // 데이터 로더 파트도 간소화 가능? // 기존 구조 유지하면서 가능할 거 같은데
    # 대체 가능한 부분, transpose, 데이터로더의 입력도, 토이파일 통해서 확인할수 있지않나?
    dataset = CreateMLBYoutubeDataset(split_file=SPLIT_FILE_PATH, mode="training", transform=test_transform, root_path=ROOT_PATH, save_dir=SAVE_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True) # num_workers = mp // pin_memory = automatic memory allocate

    validation_dataset = CreateMLBYoutubeDataset(split_file=SPLIT_FILE_PATH, mode="testing", transform=test_transform, root_path=ROOT_PATH, save_dir=SAVE_DIR)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    datasets = {'train': dataset, 'test': validation_dataset}
    dataloaders = {'train': dataloader, 'test': validation_dataloader}
    dataset_size = {'train': len(dataset), 'test': len(validation_dataset)}

    print(dataset_size)
    """################################################################### """

    # 외부 .py 파일에서 불러오기 위해서는 아래 코드 필요
    # model = Residual_network(num_classes=8)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # Experiment implementation
    criterion = nn.CrossEntropyLoss()

    if FE_MODE is 1:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # for interpreter
    num_epoches = 25
    device = DEVICE

    # train_model(model, criterion, optimizer, scheduler, num_epoches, dataloaders, device, dataset_size):
    model_pt = train_model(model, criterion, optimizer, exp_lr_schedular, 25, dataloaders, DEVICE, dataset_size)






    """    
    > 그럼 어떻게 접근한다?
        - 우선은 i3d를 어떻게 학습시켰는지 파악, 데이터가 들어가는 방식에 따라서 모델의 입력이 들어가는 방식이 달라질테니..
        - 
    """


if __name__ == '__main__':
    main()


