from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange
import torchvision.transforms as transforms
import csv

from dataset import uda_data
from model import DANN

def main(args):
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    if args.target == "svhn":
        test_datas = uda_data(args.infer_path, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize((0.5), (0.5)),
        ])
        test_datas = uda_data(args.infer_path, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(test_datas, batch_size=args.batch_size, shuffle=False)

    model = DANN().to(device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()
    filename_list = []
    label_list = []
    with torch.no_grad():
        for i, (filename, data) in enumerate(test_loader):
            data = data.to(device)

            pred, _ = model(data, 0)
            max_ind = torch.argmax(pred, dim=-1).tolist()
            filename_list += filename 
            label_list += max_ind
       
    with open(args.pred_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        
        writer.writerow(["image_name", "label"])
        for i in range(len(filename_list)):
            writer.writerow([filename_list[i], label_list[i]])


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--target", type=str, default="svhn")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--infer_path",
        type=Path,
        help="path to infer file.",
        default="./hw2_data/hw2_data/digits/svhn/data",
    )

    # model
    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="svhn.pt",
    )

    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Directory to soutput file.",
        default="qq.csv",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)