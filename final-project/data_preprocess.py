import pandas as pd
import glob, os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

root = "./dlcv-final-problem1-talking-to-me/student_data/student_data/"
video_frame_root = './dlcv-final-problem1-talking-to-me/student_data/student_data/videos_frame'
split = 'test' # train
video_seg = sorted(glob.glob(os.path.join(root, split, 'seg', '*.csv')))
# video_bbox = sorted(glob.glob(os.path.join(root, split, 'bbox', '*.csv')))
Path(os.path.join(root, f'crop_{split}')).mkdir(parents=True, exist_ok=True)

for _, seg in tqdm(enumerate(video_seg)):
    video_id = seg.split('/')[-1].split('_')[0]
    # print("video_id", video_id)
    
    bbox = os.path.join(root, split, 'bbox', f'{video_id}_bbox.csv')
    bbox_csv = pd.read_csv(bbox)
    
    seg_csv = pd.read_csv(seg)
    for i in tqdm(range(len(seg_csv))):
        data = seg_csv.iloc[i]
        start = data['start_frame']
        end = data['end_frame']
        person_id = data['person_id']
        ttm = 2
        if split == 'train':
            ttm = data['ttm']   
        # print("s", start)
        # print("e", end)
        dir_name = os.path.join(root, f'crop_{split}', f'{video_id}_{person_id}_{start}_{end}_{ttm}')
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        for j in range(start, end+1):
            row = bbox_csv[bbox_csv['frame_id'] == j]
            row = row[row['person_id'] == person_id]
            # x1 = float(row['x1'])
            # y1 = float(row['y1'])
            # x2 = float(row['x2'])
            # y2 = float(row['y2'])
            x1 = int(row['x1'])
            y1 = int(row['y1'])
            x2 = int(row['x2'])
            y2 = int(row['y2'])
            
            if x1 == -1 and y1 == -1 and x2 == -1 and y2 == -1:
                continue
            
            img = cv2.imread(os.path.join(video_frame_root, video_id, f'{"%04d" % j}.png'))
            cropped = img[y1:y2, x1:x2]
            if np.any(cropped) == False:
                continue
            cv2.imwrite(os.path.join(dir_name, f'{"%04d" % j}.png'), cropped)
            
            