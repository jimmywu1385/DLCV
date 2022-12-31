import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import torch
from tqdm import trange
import torchvision.transforms as transforms
import numpy as np

from dataset import uda_data
from model import DANN

mnistm_root = os.path.join("hw2_data", "hw2_data", "digits", "mnistm")
svhn_root = os.path.join("hw2_data", "hw2_data", "digits", "svhn")
usps_root = os.path.join("hw2_data", "hw2_data", "digits", "usps")

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
        transforms.Normalize((0.5), (0.5)),
    ])

    src_datas = uda_data(os.path.join(mnistm_root, "data"), os.path.join(mnistm_root, "train.csv"), transform)
    if args.target == "svhn":
        val_datas = uda_data(os.path.join(svhn_root, "data"), os.path.join(svhn_root, "val.csv"), transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize((0.5), (0.5)),
        ])
        val_datas = uda_data(os.path.join(usps_root, "data"), os.path.join(usps_root, "val.csv"), transform)
    
    src_loader = torch.utils.data.DataLoader(src_datas, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_datas, batch_size=args.batch_size, shuffle=False)

    model = DANN().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()

        train_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (src_label, src_data) in enumerate(src_loader):          
            src_label = src_label.to(device)
            src_data = src_data.to(device)

            optimizer.zero_grad()

            src_class, _ = model(src_data, alpha=0)
            src_class_loss = criterion(src_class, src_label)

            loss = src_class_loss

            loss.backward()
            optimizer.step()            

            max_ind = torch.argmax(src_class, dim=-1)
            correct += (max_ind == src_label).sum().item()
            train_loss += loss.item()
            total += src_label.size(0)
            if (i+1) % 50 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f} acc: {correct / total:.5f}")

        print("------eval start------\n")

        model.eval()
        eval_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (label, data) in enumerate(val_loader):
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                pred, _ = model(data, 0)
                loss = criterion(pred, label)
            
            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            eval_loss += loss.item()
            total += label.size(0)
        
        print(f"epoch : {epoch + 1}, loss: {eval_loss / (i+1):.5f} acc: {correct / total:.5f}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--random_seed", type=int, default=123)

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--target", type=str, default="svhn")

    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)