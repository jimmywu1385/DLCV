from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import torch
from tqdm import trange
import torchvision.transforms as transforms
from torchvision.utils  import save_image, make_grid

from model import condition_UNet

def val_sample(data, alpha_sqr, minus_alpha_accum_sqr, beta, model, label, t): 
    output = model(data, t, label) 
    out = alpha_sqr.gather(-1, t)
    alpha_sqr_t = out.reshape(data.size(0), 1, 1, 1) 
    out = minus_alpha_accum_sqr.gather(-1, t)
    minus_alpha_accum_sqr_t = out.reshape(data.size(0), 1, 1, 1)
    out = beta.gather(-1, t)
    beta_t = out.reshape(data.size(0), 1, 1, 1)
  
    output = 1.0 / alpha_sqr_t * (data - beta_t / minus_alpha_accum_sqr_t * output)
    if t[0] >= 1:
        noise = torch.randn_like(output).to(output.device)         
        return output + torch.sqrt(beta_t) * noise
    else:
        return output 

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
        transforms.Resize(28),
    ])

    # alpha beta
    beta = torch.linspace(args.beta_start, args.beta_end, args.timestep).to(device)
    alpha = 1.0 - beta
    alpha_sqr = torch.sqrt(alpha)
    alpha_accum = torch.cumprod(alpha, dim=0)
    minus_alpha_accum_sqr = torch.sqrt(1.0 - alpha_accum)

    model = condition_UNet(args.hidden_size, 10).to(device)
    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    
    model.eval()
    with torch.no_grad():
        grid = []
        for i in trange(10, desc="digit"):
            data = torch.randn(10, 3, 32, 32).to(device)
            for j in reversed(range(args.timestep)):
                label = torch.full((data.size(0),), i).to(device)
                time = torch.full((data.size(0),), j).to(device)
                data = val_sample(data, alpha_sqr, minus_alpha_accum_sqr, beta, model, label, time)
                if ((j+1) % 10 == 0 or j+1 == args.timestep or j == 0) and i == 0:
                    save_image(i_transform(data[0]), Path(f"0_step{j+1:0>3d}.png"))
            grid.append(data)
            
        grid = torch.cat(grid, 0)
        i_transform(grid)
        grid_img = make_grid(grid, nrow=10)
        save_image(grid_img, Path("d_grid.png"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--random_seed", type=int, default=123)

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    # beta alpha
    parser.add_argument("--beta_start", type=int, default=0.0001)
    parser.add_argument("--beta_end", type=int, default=0.02)
    parser.add_argument("--timestep", type=int, default=400)


    parser.add_argument("--hidden_size", type=int, default=256)

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--infer_dir",
        type=Path,
        help="Directory to infer the model file.",
        default="./infer_d/",
    )
    # model
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="diffusion.pt",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.infer_dir.mkdir(parents=True, exist_ok=True)
    main(args)