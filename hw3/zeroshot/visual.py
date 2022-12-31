import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

import torch
import clip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import time

root = os.path.join("hw3_data", "p1_data")

def main(args):
    random.seed(time.time())

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    model, transform = clip.load(args.model_name, device=device)

    origin_image = []
    images = []
    filelist = [filename for filename in os.listdir(os.path.join(root, "val"))]
    random.shuffle(filelist)
    for i, filename in enumerate(filelist):
        if i >= 1:
            break
        image = Image.open(os.path.join(root, "val", filename)).convert("RGB") 
        origin_image.append(image)
        images.append(transform(image))
    images = torch.tensor(np.stack(images)).to(device)

    with open(os.path.join(root, "id2label.json"), "r") as f:
        dic = json.load(f)
    prompt_text = [args.prompt.replace("[class]", val) for _, val in dic.items()]
    prompt = clip.tokenize(prompt_text).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(prompt).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)       
                
    plt.figure(figsize=(9, 9))

    for i, image in enumerate(origin_image):
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, 100 * top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)  
        plt.xlim([0, 100])
        plt.yticks(y, [prompt_text[index] for index in top_labels[i].numpy()])
        plt.xlabel(f"correct probability : {100 * top_probs[i][0]:.2f}%")

    plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("fig1.png")
            

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

    # model
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--prompt", type=str, default="This is a photo of [class]")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)