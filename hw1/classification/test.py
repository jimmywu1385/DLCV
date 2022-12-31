from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv

import torch
import torchvision.transforms as transforms

from dataset import Cls_data
from model import resnet

def main(args):
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    testdata = Cls_data(args.data_dir, transform, test=True)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    model = resnet(test=True).to(device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    filename_list = []
    label_list = []
    with torch.no_grad():
        for i, (filename, data) in enumerate(testloader):
            data = data.to(device)

            pred = model(data)
            max_ind = torch.argmax(pred, dim=-1).tolist()
            filename_list += filename 
            label_list += max_ind
       
    with open(args.pred_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        
        writer.writerow(["filename", "label"])
        for i in range(len(filename_list)):
            writer.writerow([filename_list[i], label_list[i]])

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to test file.",
        default="./hw1_data/hw1_data/p1_data/val_50",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Directory to soutput file.",
        default="qq.csv",
    )

    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="baseline_cls.pt",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)