import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import torch
from tqdm import trange
import torchvision.transforms as transforms
from torchvision.utils  import save_image

from dataset import dif_data
from model import condition_UNet

root = os.path.join("hw2_data", "hw2_data", "digits", "mnistm", "data")

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

def train_sample(data, alpha_accum_sqr, minus_alpha_accum_sqr, noise, t):
    out = alpha_accum_sqr.gather(-1, t)
    alpha_accum_sqr_t = out.reshape(data.size(0), 1, 1, 1)
    out = minus_alpha_accum_sqr.gather(-1, t)
    minus_alpha_accum_sqr_t = out.reshape(data.size(0), 1, 1, 1)
    return alpha_accum_sqr_t * data + minus_alpha_accum_sqr_t * noise

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

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    i_transform = transforms.Compose([
        transforms.Resize(28),
    ])

    label_path = os.path.join("hw2_data", "hw2_data", "digits", "mnistm", "train.csv")
    traindata = dif_data(root, label_path, transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)

    # alpha beta
    beta = torch.linspace(args.beta_start, args.beta_end, args.timestep).to(device)
    alpha = 1.0 - beta
    alpha_sqr = torch.sqrt(alpha)
    alpha_accum = torch.cumprod(alpha, dim=0)
    alpha_accum_sqr = torch.sqrt(alpha_accum)
    minus_alpha_accum_sqr = torch.sqrt(1.0 - alpha_accum)

    model = condition_UNet(args.hidden_size, 10).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        for i, (label, data) in enumerate(trainloader):
            label = label.to(device)
            data = data.to(device)
            noise = torch.randn_like(data).to(device)
            time = torch.randint(0, args.timestep, (len(data), )).to(device)
            noisy_data = train_sample(data, alpha_accum_sqr, minus_alpha_accum_sqr, noise, time)

            optimizer.zero_grad()

            pred = model(noisy_data, time, label)
            loss = criterion(pred, noise)
            loss.backward()  

            optimizer.step()
            if (i+1) % 100 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {loss.item():.5f}")

        print("------valid start------")  
        model.eval()
        with torch.no_grad():
            for i in trange(10, desc="digit"):
                data = torch.randn(100, 3, 32, 32).to(device)
                for j in reversed(range(args.timestep)):
                    label = torch.full((data.size(0),), i).to(device)
                    time = torch.full((data.size(0),), j).to(device)
                    data = val_sample(data, alpha_sqr, minus_alpha_accum_sqr, beta, model, label, time)
                for j, img in enumerate(data):
                    save_image(i_transform(img), args.infer_dir / Path(f"{i}_{j:0>3d}.png"))

        script = "python digit_classifier.py --folder " + str(args.infer_dir)
        with os.popen(script, "r") as f:
            acc = float(f.read().split()[6])
            print(f"epoch : {epoch + 1}, acc: {acc:.5f}")

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
    parser.add_argument("--batch_size", type=int, default=128)

    # beta alpha
    parser.add_argument("--beta_start", type=int, default=0.0001)
    parser.add_argument("--beta_end", type=int, default=0.02)
    parser.add_argument("--timestep", type=int, default=400)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

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

    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.infer_dir.mkdir(parents=True, exist_ok=True)
    main(args)