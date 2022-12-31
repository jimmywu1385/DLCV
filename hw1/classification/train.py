import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange
import torchvision.transforms as transforms

from dataset import Cls_data
from model import resnet, my_cnn

root = os.path.join("hw1_data", "hw1_data", "p1_data")
data_dir = {
        "train" : "train_50",
        "val" : "val_50",
    }

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
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    traindata = Cls_data(os.path.join(root, data_dir["train"]), transform)
    valdata = Cls_data(os.path.join(root, data_dir["val"]), transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False)

    #model = resnet().to(device)
    model = my_cnn().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (label, data) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            train_loss += loss.item()
            total += label.size(0)
            if i % 50 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f} acc: {correct / total:.5f}")

        print("------eval start------\n")

        model.eval()
        eval_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (label, data) in enumerate(valloader):
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, label)
            
            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            eval_loss += loss.item()
            total += label.size(0)
            if i % 50 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f} acc: {correct / total:.5f}")

        if args.visual:
            if epoch == 1 or epoch == args.num_epoch // 2:
                torch.save(model.state_dict(), args.ckpt_dir / (args.model_name.name[:-3]+str(epoch)+".pt"))       

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
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--step_size", type=float, default=1000)
    parser.add_argument("--gamma", type=float, default=1)

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
        default="cls.pt",
    )

    parser.add_argument("--num_epoch", type=int, default=20)

    parser.add_argument("--visual", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)