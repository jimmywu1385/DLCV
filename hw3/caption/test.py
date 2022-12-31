import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer

from dataset import caption_data
from model import image_caption

def main(args):

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = Tokenizer.from_file(os.path.join(args.ckpt_dir, "caption_tokenizer.json"))

    model = image_caption(args.name, True, max_len=args.max_len).to(device)


    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    valdata = caption_data(args.data_dir, tokenizer, args.max_len, transform=transform)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=1, shuffle=False)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    json_out = {}
    for _, (file, image) in enumerate(valloader):
        file = file[0].replace(".jpg", "")
        image = image.to(device)
        caption_in = torch.full((1, 1), 2).to(device)

        caption = model.generate(image, caption_in)
        caption = tokenizer.decode(caption.squeeze().tolist())

        json_out[file] = caption

    with open(args.pred_file, "w") as f:        
        json.dump(json_out, f, ensure_ascii=False, indent=4)     


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
        "--data_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./hw3_data/p2_data/images/val",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Directory to json file.",
        default="./out.json",
    )

    parser.add_argument("--max_len", type=int, default=55)
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="caption.pt",
    )
    parser.add_argument("--name", type=str, help="model name.", default="vit_large_patch14_224_clip_laion2b")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)