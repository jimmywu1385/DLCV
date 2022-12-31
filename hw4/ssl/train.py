import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange
import torchvision.transforms as transforms
import torchvision.models as models

from dataset import Cls_data
from model import resnet

root = os.path.join("hw4_data", "office")

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
    valdata = Cls_data(os.path.join(root, "val"), os.path.join(root, "val.csv"), transform, traindata.label_dic)
    print(traindata.label_dic)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False)

    backbone = None
    if args.backbone_model_name != None:
        backbone = models.resnet50(weights=None).to(device)
        mckpt = torch.load(args.ckpt_dir / args.backbone_model_name)
        backbone.load_state_dict(mckpt)
    model = resnet(65, backbone=backbone, freeze=args.freeze).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (_, _, data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
    
            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            train_loss += loss.item()
            total += label.size(0)
            if i % 10 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f} acc: {correct / total:.5f}")

        print("------eval start------\n")

        model.eval()
        eval_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (_, _, data, label) in enumerate(valloader):
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, label)
        
            max_ind = torch.argmax(pred, dim=-1)
            correct += (max_ind == label).sum().item()
            eval_loss += loss.item()
            total += label.size(0)
            if i % 10 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f} acc: {correct / total:.5f}")       
        print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f} acc: {correct / total:.5f}")       
        if best < (correct / total):
            best = correct / total
            torch.save(model.state_dict(), args.ckpt_dir / Path("best.pt"))

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
    parser.add_argument("--batch_size", type=int, default=64)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)

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
    parser.add_argument(
        "--backbone_model_name",
        type=Path,
        help="model name.",
        default=None,
    )

    parser.add_argument("--num_epoch", type=int, default=50)

    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)