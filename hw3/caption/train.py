import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

import torch
from tqdm import trange
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer

from dataset import caption_data
from model import image_caption
from eval_matric import evaluate

root = os.path.join("hw3_data", "p2_data")

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = Tokenizer.from_file("./hw3_data/caption_tokenizer.json")

    model = image_caption(args.name, max_len=args.max_len).to(device)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    traindata = caption_data(os.path.join(root, "images", "train"), tokenizer, args.max_len, os.path.join(root, "train.json"), transform)
    valdata = caption_data(os.path.join(root, "images", "val"), tokenizer, args.max_len, transform=transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=1, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for i, (caption, image) in enumerate(trainloader):
            caption_in = caption["caption_in"].to(device)
            caption_out = caption["caption_out"].to(device)
            caption_mask = caption["mask"].to(device)
            image = image.to(device)

            optimizer.zero_grad()
            pred = model(image, caption_in, caption_mask)
            pred = model.predictor(pred)
            token = (caption_out != 0).sum()
            loss = criterion(pred.contiguous().view(-1, pred.size(-1)), caption_out.contiguous().view(-1)) / token
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (i+1) % 100 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss:.5f}")
                train_loss = 0
        
        print("---eval start---")
        model.eval()
        json_out = {}
        for i, (file, image) in enumerate(valloader):
            file = file[0].replace(".jpg", "")
            image = image.to(device)
            caption_in = torch.full((1, 1), 2).to(device)

            caption = model.generate(image, caption_in)
            caption = tokenizer.decode(caption.squeeze().tolist())

            json_out[file] = caption

        with open(args.pred_file, "w") as f:        
            json.dump(json_out, f, ensure_ascii=False, indent=4)     
        
        evaluate(args)
        
    torch.save(model.state_dict(), args.ckpt_dir / args.model_name)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--random_seed", type=int, default=123)

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Directory to json file.",
        default="./out.json",
    )
    parser.add_argument("--annotation_file", type=Path, default="./hw3_data/p2_data/val.json")
    parser.add_argument("--images_root", type=Path, default="./hw3_data/p2_data/images/val")

    parser.add_argument("--max_len", type=int, default=55)
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="caption.pt",
    )
    parser.add_argument("--name", type=str, help="model name.", default="vit_large_patch14_224_clip_laion2b")

    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)