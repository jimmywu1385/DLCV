1. Data Preprocessing

## Original Datasets Structure
```
/dlcv-final-problem1-talking-to-me
    /student_data
        /student_data
            /train
            /test
            /videos
```

### Image
- First turn the videos to frames and save them in "./dlcv-final-problem1-talking-to-me/student_data/student_data/videos_frame".
```
python video_preprocess.py
```
- Then crop the face according to bbox. Edit the variable split = "train" or "test" at line 10 in data_prerpocess.py. The result will be saved in "./dlcv-final-problem1-talking-to-me/student_data/student_data/crop_{split}".
```
python data_preprocess.py
```

### Audio
- Save the audio file of video in "./dlcv-final-problem1-talking-to-me/student_data/student_data/audios".
```
python preprocess_audio.py
```

## Datasets Configuration
```
/dlcv-final-problem1-talking-to-me
    /student_data
        /student_data
            /train
            /test
            /videos
            /audios
            /video_frames
            /crop_train
            /crop_test
```

2. Training(or skip this step and use our checkpoint below directly)
```
python train.py
```
- After running the script, you will get checkpoints saved in "ckpt/{epoch}.pt".
Checkpoint link
- https://drive.google.com/file/d/1qP5pDC40kGjfz0zzHIyNkoNX6gFYdIeL/view?usp=sharing 


3. Testing
```
python test.py --ckpt_dir <path/to/ckpt/directory/> --model_name <checkpoint_name>
```
e.g., python test.py --ckpt_dir ./ckpt --model_name 1.pt

- After runnig the script, you will get a "output.csv" at the root directory.