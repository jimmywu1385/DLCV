import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv

import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from transformers import Wav2Vec2FeatureExtractor

from dataset import video_audio_data
from model import waveLM_Resnet152


root = os.path.join("dlcv-final-problem1-talking-to-me", "student_data", "student_data")

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    data = video_audio_data(os.path.join(root, "crop_test"), os.path.join(root, "audios"), 
                            transform, 5, 8 * 16000, True, processor,
                        )

    testloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)

    model = waveLM_Resnet152().to(device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    ID = []
    Predicted = []

    model.eval()
    for i, (video_length, video, audio, filename) in enumerate(tqdm(testloader)):
        video = video.to(device)
        audio["input_values"] = audio.input_values.squeeze(1)
        audio["attention_mask"] = audio.attention_mask.squeeze(1)
        audio = audio.to(device)


        with torch.no_grad():
            pred = model(audio, video, video_length)

        max_ind = torch.argmax(pred, dim=-1).tolist()
        Predicted += max_ind
        ID += filename

    with open(args.pred_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        
        writer.writerow(["ID", "Predicted"])
        for i in range(len(ID)):
            writer.writerow([ID[i], Predicted[i]])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--random_seed", type=int, default=123)

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--video_maxlen", type=int, help="number of frame should be multiplier of 16", default=160)

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./",
    )

    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="1.pt",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        default="output.csv",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)