import os
from pathlib import Path
from moviepy.editor import VideoFileClip

root = "./dlcv-final-problem1-talking-to-me/student_data/student_data"
Path(os.path.join(root, "audios")).mkdir(parents=True, exist_ok=True)
for i in os.listdir(os.path.join(root, "videos")):
    video = VideoFileClip(os.path.join(root, "videos", i))
    audio = video.audio
    audio.write_audiofile(os.path.join(root, "audios", i.replace(".mp4", ".wav")))
