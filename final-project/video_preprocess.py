import cv2
import os
from pathlib import Path
from tqdm import tqdm 

video_root = './dlcv-final-problem1-talking-to-me/student_data/student_data/videos'
filenames = os.listdir(video_root) # print(len(filenames))
video_frame_root = './dlcv-final-problem1-talking-to-me/student_data/student_data/videos_frame'
Path(video_frame_root).mkdir(parents=True, exist_ok=True)

for j, video_name in tqdm(enumerate(filenames)):
    
    video_id = video_name.split('.')[0] 
    
    Path(os.path.join(video_frame_root, video_id)).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(os.path.join(video_root, video_name))
    # 5*60*3=9000
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(num_frames)):
        retval, frame = cap.read()
        if retval:
            cv2.imwrite(os.path.join(video_frame_root, video_id, f'{"%04d" % i}.png'), frame) 