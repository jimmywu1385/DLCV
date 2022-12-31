import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange
import torchvision.transforms as transforms
from byol_pytorch import BYOL
from torchvision import models

from dataset import Cls_data

root = os.path.join("hw4_data", "mini")

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    traindata = Cls_data(os.path.join(root, "train"), os.path.join(root, "train.csv"), transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)

    model = models.resnet50(weights=None).to(device)

    learner = BYOL(
        model,
        image_size=128,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    ).to(device)
    
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)

    best_lost = 10.
    best_epoch = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for i, (_, _, data, _) in enumerate(trainloader):
            data = data.to(device)

            loss = learner(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            learner.update_moving_average()

            train_loss += loss.item()
            if (i+1) % 50 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f}")   

        if best_lost > (train_loss / (i+1)):
            best_lost = (train_loss / (i+1))
            best_epoch = epoch
            torch.save(model.state_dict(), args.ckpt_dir / Path("best.pt"))

    torch.save(model.state_dict(), args.ckpt_dir / args.model_name)
    print(best_epoch)

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

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="pretrain.pt",
    )

    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)