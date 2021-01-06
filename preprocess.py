import os
import re
import cv2
import numpy as np
import multiprocessing

"""
# 21-01-06 추후 수정사항
1. 각 영상별 접근 경로 수정, 현재는 'one_img_path'로 설정되어 있으니, 반복문 추가하여, 돌아가며 영상에 대한 Sampling 수행할 수 있도록 수정.
2. 적절한 샘플링 단계 설정 
    - 아래 설명 참조
"""

DATA_ROOT_PATH = '/home/gon/Desktop/bb_extracted_frames'
DATA_DUMP_PATH = '/home/gon/Desktop/210106_centercrop'

vid = os.listdir(DATA_ROOT_PATH)
one_img_path = os.path.join(DATA_ROOT_PATH, vid[0]) # 수정사항, 이 경로가 돌아가게..


def natural_sort(l):
    """
    Sort String Data, more Naturally
    convert list a = ['1', '11', '2', '21'] >> ['1','2','11','21]

    :param l:
    :return sorted(l):
    """
    convert = lambda text: int(text) if  text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


video_list = os.listdir(one_img_path)
sorted_video_list = natural_sort(video_list)


for k in range(len(sorted_video_list)):
    img = cv2.imread(os.path.join(one_img_path, sorted_video_list[k]))
    # Image resize
    resize_with_ratio = cv2.resize(img, dsize=(0, 0), fx=0.311, fy=0.311, interpolation=cv2.INTER_AREA) # (398, 224, 3)

    # Center crop
    rh, rw, rc = resize_with_ratio.shape

    size = (224, 224)
    th, tw = size

    i = int(np.round((rh - th) / 2.))
    j = int(np.round((rw - tw) / 2.))

    cropped_img = resize_with_ratio[i:i + th, j:j + tw]

    if k % 8 == 0: # 매 8 frame마다 저장
        cv2.imwrite("%s/%d.jpg" % (DATA_DUMP_PATH, k), cropped_img)






def crop_func_backup_210105():
    img = cv2.imread(os.path.join(one_img_path, video_list[0]))
    h, w, c = img.shape

    size = (360, 720)

    """
    #21-01-05
    Center crop - 진짜 이미지를 자르는게 아니다..

    """

    th, tw = size

    i = int(np.round((h - th) / 2.))
    j = int(np.round((w - tw) / 2.))

    croped_img = img[i:i + th, j:j + tw]
    print(croped_img.shape)

    cv2.imshow("cropped_img", croped_img)
    cv2.waitKey()
