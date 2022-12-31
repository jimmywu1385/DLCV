import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange, tqdm
import torchvision.transforms as transforms
from torch.utils.data import random_split
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
    traindata = video_audio_data(os.path.join(root, "crop_train"), os.path.join(root, "audios"), 
                            transform, 5, 8 * 16000, False, processor,
                        )
    #traindata, valdata = random_split(data, [int(len(data) * 0.9), len(data) - int(len(data) * 0.9)])

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    #valloader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False)

    model = waveLM_Resnet152().to(device)
    # model.load_state_dict(torch.load('./ckpt/1.pt'))
    # model.train().to(device)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (video_length, video, audio, label) in enumerate(tqdm(trainloader)):
            video = video.to(device)
            audio["input_values"] = audio.input_values.squeeze(1)
            audio["attention_mask"] = audio.attention_mask.squeeze(1)
            audio = audio.to(device)
            label = label.to(device)

            pred = model(audio, video, video_length)
            loss = criterion(pred, label)
            loss /= args.accum_size
            loss.backward()

            if (i+1) % args.accum_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            train_loss += loss.item()
            total += label.size(0)
            if (i+1) % 10 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f} acc: {correct / total:.5f}")

        '''
        print("------eval start------\n")
    
        model.eval()
        eval_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (video_length, video, audio, label) in enumerate(tqdm(valloader)):
            video = video.to(device)
            audio["input_values"] = audio.input_values.squeeze(1)
            audio["attention_mask"] = audio.attention_mask.squeeze(1)
            audio = audio.to(device)
            label = label.to(device)
            
            with torch.no_grad():
                pred = model(audio, video, video_length)
                loss = criterion(pred, label)
        
            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            eval_loss += loss.item()
            total += label.size(0)
            if (i+1) % 100 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f} acc: {correct / total:.5f}")
        '''
        torch.save(model.state_dict(), args.ckpt_dir / Path(f"{epoch}.pt"))      
    
    torch.save(model.state_dict(), args.ckpt_dir / args.model_name)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_size", type=int, default=4)
    parser.add_argument("--video_maxlen", type=int, help="number of frame should be multiplier of 16", default=160)

    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="cls.pt",
    )

    parser.add_argument("--num_epoch", type=int, default=5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)