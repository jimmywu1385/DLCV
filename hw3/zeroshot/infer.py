import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

import torch
import clip
import csv

from dataset import CLIP_data

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    model, transform = clip.load(args.model_name, device=device)

    valdata = CLIP_data(args.data_path, transform)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False)
    
    with open(args.json_path, "r") as f:
        dic = json.load(f)
    prompt = [args.prompt.replace("[class]", val) for _, val in dic.items()]
    prompt = clip.tokenize(prompt)
    
    correct = 0
    total = 0
    filenames = []
    labels = []
    with torch.no_grad():
        for i, (file, image) in enumerate(valloader):
            image = image.to(device)
            text = prompt.to(device)
            
            image_feature = model.encode_image(image)
            text_feature = model.encode_text(text)

            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_feature @ text_feature.T).softmax(dim=-1)
            max_ind = torch.argmax(similarity, dim=-1)
            
            if args.eval:
                label = torch.tensor([int(i.split("_")[0]) for i in file]).to(device)
                correct += (max_ind == label).sum().item()
                total += image.size(0)
                print(f"step: {i}, acc: {correct/total}")            
            
            filenames += file
            labels += max_ind.cpu().tolist()
    
    with open(args.pred_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
        
            writer.writerow(["filename", "label"])
            for i in range(len(filenames)):
                writer.writerow([filenames[i], labels[i]])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--random_seed", type=int, default=123)

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    # eval
    parser.add_argument("--eval", help="show val acc", action="store_true")
    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # model
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--prompt", type=str, default="The photo is [class]")

    parser.add_argument(
        "--data_path",
        type=Path,
        default="./hw3_data/p1_data/val",
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default="./hw3_data/p1_data/id2label.json",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        default="qq.csv",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)