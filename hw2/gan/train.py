import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import torch
from tqdm import trange
import torchvision.transforms as transforms
from torchvision.utils  import save_image
import csv

from dataset import gan_data
from model import Discriminator, Generator, b_Discriminator, b_Generator

import sys
sys.path.append("C:\\Users\\W10\\Desktop\\hw2-jimmywu1385")
from face_recog import face_recog
from pytorch_fid.fid_score import calculate_fid_given_paths

root = os.path.join("hw2_data", "hw2_data", "face")

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

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
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
    ])
    i_transform = transforms.Compose([
        transforms.Normalize([0.0, 0.0, 0.0],[1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize([-0.5, -0.5, -0.5],[1.0, 1.0, 1.0]),
    ])

    traindata = gan_data(os.path.join(root, "train"), transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)

    g_model = Generator(args.g_in, args.g_hidden, args.g_out_d_in).to(device)
    d_model = Discriminator(args.g_out_d_in, args.d_hidden).to(device)
    #g_model = b_Generator(args.g_in, args.g_hidden, args.g_out_d_in).to(device)
    #d_model = b_Discriminator(args.g_out_d_in, args.d_hidden).to(device)
    g_model.apply(weights_init)
    d_model.apply(weights_init)

    criterion = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    with open("./valid.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')      
        writer.writerow(["epoch", "fid", "acc"])
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        g_model.train()
        d_model.train()

        for i, (_, data) in enumerate(trainloader):
            data = data.to(device)
            t_label = torch.full((data.size(0), ), 1.0).to(device)
            f_label = torch.full((data.size(0), ), 0.0).to(device)
            noise = torch.randn(data.size(0), args.g_in, 1, 1).to(device)

            # train discriminator
            d_optimizer.zero_grad()

            pred = d_model(data).squeeze()
            d_true_loss = criterion(pred, t_label)
            d_true_loss.backward()

            pred = d_model(g_model(noise).detach()).squeeze()
            d_false_loss = criterion(pred, f_label)
            d_false_loss.backward()

            d_optimizer.step()

            # train generator
            g_optimizer.zero_grad()

            pred = d_model(g_model(noise)).squeeze()
            g_loss = criterion(pred, t_label)
            g_loss.backward()  

            g_optimizer.step()

            if (i+1) % 5 == 0:
                print(f"epoch : {epoch + 1} step {i + 1} discriminator loss: {d_true_loss+d_false_loss :.5f} generator loss: {g_loss :.5f}")
        
        # eval
        g_model.eval()
        noise = torch.randn(1000, args.g_in, 1, 1).to(device)
        pred = g_model(noise)
        for i, data in enumerate(pred):
            save_image(i_transform(data), args.infer_dir / Path(f"{i}.png"))

        fid = calculate_fid_given_paths([str(args.infer_dir), str(os.path.join(root, "val"))], 50, device, 2048, 1)
        acc = face_recog(args.infer_dir)

        with open("./valid.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([epoch, fid, acc])

    torch.save(g_model.state_dict(), args.ckpt_dir / args.model_name)

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

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-5)

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
        default="./infer/",
    )
    # model
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="gan_improve.pt",
    )
    parser.add_argument("--g_in", type=int, default=100)
    parser.add_argument("--g_hidden", type=int, default=64)
    parser.add_argument("--g_out_d_in", type=int, default=3)
    parser.add_argument("--d_hidden", type=int, default=64)

    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.infer_dir.mkdir(parents=True, exist_ok=True)
    main(args)