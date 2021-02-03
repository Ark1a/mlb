import os
import numpy as np
import cv2
import multiprocessing
import datetime

run_start = datetime.datetime.now()

DATA_ROOT_PATH = '/home/gon/Desktop/bb_segment_video'
DUMP_PATH = '/home/gon/Desktop/sampled_video'
FRAME_RATE = int(3) # 3fps

video_list = os.listdir(DATA_ROOT_PATH)


def frame_extractor(vid):
    current_vid_path = os.path.join(DATA_ROOT_PATH, vid)
    video = cv2.VideoCapture(current_vid_path)

    frame_width, frame_height = int(video.get(3)), int(video.get(4))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter('/%s/%s' % (DUMP_PATH, vid), cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, size)

    i = 0
    while(True):
        ret, frame = video.read()

        if ret is False:
            break

        if i % 15 is 0:
            result.write(frame)

        i += 1

    video.release()
    result.release()

    cv2.destroyAllWindows()


pool = multiprocessing.Pool(processes=8)
pool.map(frame_extractor, [i for i in video_list])

run_end = datetime.datetime.now()

print("\n# Result ")
print("Started  : %s" % (run_start.strftime("%Y-%m-%d. %H:%M, %a")))
print("Finished : %s" % (run_end.strftime("%Y-%m-%d. %H:%M, %a")))
