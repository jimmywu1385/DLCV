import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import image_caption

def main(args):

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = Tokenizer.from_file(os.path.join(args.ckpt_dir, "caption_tokenizer.json"))

    model = image_caption(args.name, True, max_len=args.max_len).to(device)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    eval_img = Image.open(args.image_path).convert("RGB")
    image = transform(eval_img).unsqueeze(0).to(device)
    caption_in = torch.full((1, 1), 2).to(device)

    caption = model.generate(image, caption_in)
    attn = model.decoder.layers[-1].scores.squeeze(0)
    words = [tokenizer.decode([caption[0][i]]) for i in range(caption.size(1))]
    words[0] = "<BOS>"
    words[-1] = "<EOS>"

    fig = plt.figure(figsize=(13,13))
    column = 5
    row = int(np.ceil(len(words)/5))
    for i, (att, word) in enumerate(zip(attn, words)):
        att = att[1:].detach().cpu().numpy().reshape(7, 7)
        att = att / att.max()
        att = Image.fromarray(att).resize(eval_img.size)
        fig.add_subplot(row, column, i+1)
        plt.title(word)
        plt.imshow(eval_img)
        if i != 0:
            plt.imshow(att / np.max(att), cmap='rainbow', alpha=0.4)
        plt.axis('off')
    plt.savefig(args.save_path, bbox_inches='tight')    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        help="Directory to save the model file.",
        default="./hw3_data/p3_data/images/sheep.jpg",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Directory to save the model file.",
        default="./qq.jpg",
    )

    parser.add_argument("--max_len", type=int, default=55)
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="base.pt",
    )
    parser.add_argument("--name", type=str, help="model name.", default="vit_base_patch32_224_clip_laion2b")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)