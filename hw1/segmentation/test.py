from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from dataset import Seg_data
from model import vgg16FCN32

def main(args):
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    testdata = Seg_data(args.data_dir, transform, test=True)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    model = vgg16FCN32(args.num_class).to(device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    filename_list = []
    image_list = []
    with torch.no_grad():
        for i, (filename, data) in enumerate(testloader):
            data = data.to(device)

            pred = model(data)
            max_ind = torch.argmax(pred, dim=1).cpu().numpy()
            filename_list += filename 
            image_list.append(max_ind)
    
    image_list = np.concatenate(image_list)
    color_map = {
        0: [0, 255, 255],
        1: [255, 255, 0],
        2: [255, 0, 255],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [255, 255, 255],
        6: [0, 0, 0],
    }
    for ind in range(len(filename_list)):
        image = image_list[ind]
        out_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for i in range(args.num_class):
            out_image[image == i] = np.array(color_map[i])
        PIL_image = Image.fromarray(out_image)
        PIL_image.save(args.pred_dir / filename_list[ind].replace("jpg", "png"))

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # device
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    # data loader
    parser.add_argument("--num_class", type=int, default=7)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

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
        default="./hw1_data/hw1_data/p2_data/validation",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        help="Directory to soutput file.",
        default="./pred_image",
    )

    parser.add_argument(
        "--model_name",
        type=Path,
        help="model name.",
        default="vgg16fcn32.pt",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    main(args)