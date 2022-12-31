from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import torch
import torchvision.transforms as transforms
from torchvision.utils  import save_image, make_grid

from model import Generator, b_Generator

def set_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    i_transform = transforms.Compose([
        transforms.Normalize([0.0, 0.0, 0.0],[1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize([-0.5, -0.5, -0.5],[1.0, 1.0, 1.0]),
    ])
    
    #g_model = b_Generator(args.g_in, args.g_hidden, args.g_out_d_in).to(device)
    g_model = Generator(args.g_in, args.g_hidden, args.g_out_d_in).to(device)
    mckpt = torch.load(args.ckpt_dir / args.model_name)
    g_model.load_state_dict(mckpt)

    g_model.eval()
    noise = torch.randn(32, args.g_in, 1, 1).to(device)
    with torch.no_grad():
        pred = g_model(noise)
        grid_img = make_grid(pred, nrow=8)
        save_image(i_transform(grid_img), args.infer_name)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--random_seed", type=int, default=130)
    
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
        "--infer_name",
        type=Path,
        help="Directory to infer the model file.",
        default="grid.png",
    )

    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="gan_improve.pt",
    )
    parser.add_argument("--g_in", type=int, default=100)
    parser.add_argument("--g_hidden", type=int, default=64)
    parser.add_argument("--g_out_d_in", type=int, default=3)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)