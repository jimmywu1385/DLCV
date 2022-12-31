import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import Cls_data
from model import resnet

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

    label2id = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
    id2label = {y: x for x, y in label2id.items()}
    testdata = Cls_data(args.data_dir, args.data_file, transform, label2id)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    model = resnet(65).to(device)
    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)

    model.eval()
    ids = []
    filenames = []
    labels = []
    for i, (id, filename, data, _) in enumerate(tqdm(testloader)):
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)

        max_ind = torch.argmax(pred, dim=-1).cpu().tolist()
        labels += max_ind
        ids += id
        filenames += filename

    with open(args.pred_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        
        writer.writerow(["id", "filename", "label"])
        for i in range(len(ids)):
            writer.writerow([ids[i], filenames[i], id2label[labels[i]]])


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
        default="best.pt",
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default="./hw4_data/office/val",
    )

    parser.add_argument(
        "--data_file",
        type=Path,
        default="./hw4_data/office/val.csv",
    )

    parser.add_argument(
        "--pred_file",
        type=Path,
        default="output.csv",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)