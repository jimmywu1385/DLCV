import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange
import torchvision.transforms as transforms

from dataset import Seg_data
from model import vgg16FCN32, deeplab3

root = os.path.join("hw1_data", "hw1_data", "p2_data")
data_dir = {
        "train" : "train",
        "val" : "validation",
    }

def mean_iou_score(pred, labels):

    mean_iou = 0
    for i in range(6):
        tp_fp = torch.sum(pred == i)
        tp_fn = torch.sum(labels == i)
        tp = torch.sum((pred == i) * (labels == i))
        iou = 0
        if tp_fp + tp_fn - tp != 0:
            iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    return mean_iou

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    t_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    v_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    traindata = Seg_data(os.path.join(root, data_dir["train"]), t_transform)
    valdata = Seg_data(os.path.join(root, data_dir["val"]), v_transform)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False)

    #model = vgg16FCN32(args.num_class).to(device)
    model = deeplab3(args.num_class).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for i, (label, data) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            if (i+1) % args.accum_size == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            max_ind = torch.argmax(pred, dim=1)
            miou = mean_iou_score(max_ind, label)
            train_loss += loss.item()
            if i % 50 == 0:   
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {train_loss / (i+1):.5f} miou: {miou:.5f}")
        
        print("------eval start------\n")
    
        model.eval()
        eval_loss = 0.0
        preds = []
        labels = []
        total_batch = 0
        for i, (label, data) in enumerate(valloader):
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, label)
            
            max_ind = torch.argmax(pred, dim=1)
            preds.append(max_ind)
            labels.append(label)
            miou = mean_iou_score(max_ind, label)
            eval_loss += loss.item()
            if i % 50 == 0:
                print(f"epoch : {epoch + 1}, iter : {i + 1:5d} loss: {eval_loss / (i+1):.5f} miou: {miou:.5f}")
            total_batch = i

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        
        print(f"Total loss: {eval_loss / total_batch:.5f} Total miou: {mean_iou_score(preds, labels):.5f}\n")

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

    # number of class
    parser.add_argument("--num_class", type=int, default=7)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_size", type=int, default=2)

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
        default="vgg16fcn32.pt",
    )

    parser.add_argument("--num_epoch", type=int, default=20)

    parser.add_argument("--visual", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)