from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import Cls_data
from model import my_cnn

def main(args):
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    testdata = Cls_data(args.data_dir, transform, test=True)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    model = my_cnn().to(device)

    mckpt = torch.load(args.ckpt_dir / args.model_name)
    model.load_state_dict(mckpt)
    model.eval()

    labels = []
    features_out_hook = []
    def hook(module, fea_in, fea_out):
        fea_out = fea_out.squeeze().cpu().numpy()
        features_out_hook.append(fea_out)
        return None

    model.pool.register_forward_hook(hook)
    with torch.no_grad():
        for i, (_, data) in enumerate(testloader):
            data = data.to(device)
            pred = model(data)
            max_ind = torch.argmax(pred, dim=-1).cpu().numpy()
            labels.append(max_ind)
    feature = np.concatenate(features_out_hook)
    label = np.concatenate(labels)

    if args.transform == "tsne":
        transform = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=10)
    else:
        transform = PCA(n_components=2)
    new_feature = transform.fit_transform(feature)
    colors = cm.get_cmap("hsv", 256)
    colors = colors(np.linspace(0.05, 0.95, 50))

    
    plt.figure()
    for i in range(50):
        selected = new_feature[np.where(label == i)[0]]
        plt.scatter(selected[:, 0], selected[:, 1], color=colors[i])
    
    plt.savefig(args.output_fig)

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
        "--model_name",
        type=Path,
        help="model name.",
        default="resnet101.pt",
    )

    parser.add_argument(
        "--transform",
        type=str,
        help="transform type.",
        default="tsne",
    )

    parser.add_argument(
        "--output_fig",
        type=str,
        help="output picture name.",
        default="tsne1.png",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)