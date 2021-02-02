import os
import numpy as np
import cv2
import multiprocessing
import time

DATA_ROOT_PATH = '/home/gon/Desktop/bb_segment_video'
DUMP_PATH = '/home/gon/Desktop/bb_extracted_frames'


RESIZE_RATIO = 0.311
RESIZE_FRAME_SIZE = (224, 224)

video_list = os.listdir(DATA_ROOT_PATH)


def frame_extractor(vid):
        current_vid_path = os.path.join(DATA_ROOT_PATH, vid)
        current_vid_dump_path = os.path.join(DUMP_PATH, vid)

        try:
            if not (os.path.isdir(current_vid_dump_path)):
                os.makedirs(os.path.join(current_vid_dump_path))
        except OSError:
            print('Error : Failed to Create directory')

        cap = cv2.VideoCapture(current_vid_path)

        i = 0
        k = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # ret - 비디오를 불러오는데 성공했는가
            if ret == False:
                break

            resized_frame = cv2.resize(frame, dsize=(0, 0), fx=RESIZE_RATIO, fy=RESIZE_RATIO, interpolation=cv2.INTER_AREA)
            h, w, c = resized_frame.shape

            th, tw = RESIZE_FRAME_SIZE
            rh = int(np.round(h-th) / 2.)
            rw = int(np.round(w-tw) / 2.)

            cropped_img = resized_frame[rh:rh+th, rw:rw+tw]

            if i % 8 == 0: # 8 Frame 마다 저장
                cv2.imwrite(current_vid_dump_path + '/' + vid + '_' + str(k) + '.jpg', cropped_img)
                k += 1

            i += 1

        print("\nVid : %s, %d Frame extracted" % (vid, len(os.listdir(current_vid_dump_path))))
        cap.release()
        cv2.destroyAllWindows()


pool = multiprocessing.Pool(processes=8)
pool.map(frame_extractor, [i for i in video_list])


"""
" 21-01-07
매 8 frame 마다 sampling
Total 106,932 / 2.6GB
"""