import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchaudio
import torchvision.transforms as transforms
import random

class video_audio_data(Dataset):
    def __init__(self, videos_root, audios_root, transform, video_maxlen=1000, audio_maxlen=534000, test=False, audio_processor=None):
        self.videos_root = videos_root
        self.audios_root = audios_root

        self.video_filenames = []
        self.audio_filenames = []
        self.pids = []
        self.starts = []
        self.ends = []
        self.labels = []
        for i in os.listdir(videos_root):
            self.video_filenames.append(i)
            filename, pid, start, end, label = i.split("_")
            self.audio_filenames.append(filename + ".wav")
            self.pids.append(int(pid))
            self.starts.append(int(start))
            self.ends.append(int(end))
            self.labels.append(int(label))

        self.transform = transform
        self.video_maxlen = video_maxlen
        self.audio_maxlen = audio_maxlen
        self.test = test
        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, index):
        video_file = self.video_filenames[index]
        audio_file = self.audio_filenames[index]
        label = torch.tensor(self.labels[index])
        start = self.starts[index]
        end = self.ends[index]

        video, video_length = self.video_process(video_file)
        audio = self.audio_process(audio_file, start, end)
        if self.test:
            return video_length, video, audio, "_".join(video_file.split("_")[:-1])
        else:
            return video_length, video, audio, label

    def video_process(self, filename):
        video = []
        for i, file in enumerate(os.listdir(os.path.join(self.videos_root, filename))):

            image = Image.open(os.path.join(self.videos_root, filename, file)).convert("RGB") 
            if self.transform is not None:
                image = self.transform(image)

            video.append(image)
        if len(video) == 0:
            return torch.zeros(self.video_maxlen, 3, 64, 64), 1
        if len(video) >= self.video_maxlen:
            video = torch.stack(video)
            randomlist = random.sample(range(0, video.size(0)), self.video_maxlen)
            randomlist.sort()
            video = video[randomlist]
            return video, self.video_maxlen
        else:
            video = torch.stack(video)
            length = video.size(0)
            video = torch.cat((video, torch.zeros(self.video_maxlen-video.size(0), video.size(1), video.size(2), video.size(3))))
            return video, length

    def audio_process(self, filename, start, end):
        ori_audio, ori_sample_rate = torchaudio.load(os.path.join(self.audios_root, filename), normalize=True)
        sample_rate = 16000
        transform = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)
        audio = transform(ori_audio)
        onset = int(start / 30 * sample_rate)
        offset = int(end / 30 * sample_rate)

        crop_audio = audio[0, onset:offset]
        crop_audio = self.audio_processor(
            crop_audio, 
            sampling_rate=self.audio_processor.sampling_rate, 
            max_length=int(self.audio_processor.sampling_rate * 8.0), 
            truncation=True,
            padding='max_length',
            return_tensors="pt",
        )
        return crop_audio

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    video = "./dlcv-final-problem1-talking-to-me/student_data/student_data/crop_test"
    audio =  "./dlcv-final-problem1-talking-to-me/student_data/student_data/audios"
    data = video_audio_data(video, audio, transform)
    print(data[0][1].size())
