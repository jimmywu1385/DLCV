from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os 

from dataset import uda_data
from model import DANN

mnistm_root = os.path.join("hw2_data", "hw2_data", "digits", "mnistm")
svhn_root = os.path.join("hw2_data", "hw2_data", "digits", "svhn")
usps_root = os.path.join("hw2_data", "hw2_data", "digits", "usps")

def main(args):
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    model = DANN().to(device)

    src_datas = uda_data(os.path.join(mnistm_root, "data"), os.path.join(mnistm_root, "val.csv"), transform)
    if args.target == "svhn":
        mckpt = torch.load(args.ckpt_dir / Path("svhn.pt"))
        val_datas = uda_data(os.path.join(svhn_root, "data"), os.path.join(svhn_root, "val.csv"), transform)
    else:
        mckpt = torch.load(args.ckpt_dir / Path("usps.pt"))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize((0.5), (0.5)),
        ])
        val_datas = uda_data(os.path.join(usps_root, "data"), os.path.join(usps_root, "val.csv"), transform)

    src_loader = torch.utils.data.DataLoader(src_datas, batch_size=args.batch_size, shuffle=False)    
    val_loader = torch.utils.data.DataLoader(val_datas, batch_size=args.batch_size, shuffle=False)

    model.load_state_dict(mckpt)
    model.eval()

    labels = []
    features_out_hook = []
    def hook(module, fea_in, fea_out):
        fea_out = fea_out.squeeze().cpu().numpy()
        features_out_hook.append(fea_out)
        return None

    model.feature.register_forward_hook(hook)
    if args.domain:
        num_label = 2
        with torch.no_grad():
            for i, (_, data) in enumerate(src_loader):
                data = data.to(device)
                pred, _ = model(data, 0)
                labels.append(np.full((data.size(0),), 0))
            for i, (_, data) in enumerate(val_loader):
                data = data.to(device)
                pred, _ = model(data, 0)
                labels.append(np.full((data.size(0), ), 1))
        feature = np.concatenate(features_out_hook)
        label = np.concatenate(labels)        
    else:
        num_label = 10
        with torch.no_grad():
            for i, (label, data) in enumerate(val_loader):
                data = data.to(device)
                pred, _ = model(data, 0)
                labels.append(label.numpy())
        feature = np.concatenate(features_out_hook)
        label = np.concatenate(labels)

    transform = TSNE(n_components=2, learning_rate="auto", init="random")

    new_feature = transform.fit_transform(feature)
    colors = cm.get_cmap("hsv", 256)
    colors = colors(np.linspace(0.1, 0.9, num_label))
    
    plt.figure()
    for i in range(num_label):
        selected = new_feature[np.where(label == i)[0]]
        plt.scatter(selected[:, 0], selected[:, 1], color=colors[i], s=1)
        
    plt.ylim([-100, 100])    
    plt.xlim([-100, 100])
    plt.savefig(args.output_fig)

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
        "--output_fig",
        type=str,
        help="output picture name.",
        default="tsne1.png",
    )

    parser.add_argument("--domain", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)