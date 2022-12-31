import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image
import clip
import torch


def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CLIPScore:
    def __init__(self):
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        max_score = 0.
        max_name = None

        min_score = 1.
        min_name = None

        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")
            score = self.getCLIPScore(image, pred_caption)
            if score > max_score:
                max_score = score
                max_name = img_name
            if score < min_score:
                min_score = score
                min_name = img_name
        return {"max_score" : max_score,
                "max_name" : max_name,
                "min_score" : min_score,
                "min_name" : min_name,
            }

    def getCLIPScore(self, image, caption):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([caption]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        
        cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        return 2.5 * max(cos_sim, 0)


def evaluate(args):
    # Read data
    predictions = readJSON(args.pred_file)
    annotations = readJSON(args.annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CLIPScore
    clip_score = CLIPScore()(predictions, args.images_root)
    print(clip_score)
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pred_file", default="out.json", help="Prediction json file")
    parser.add_argument("--images_root", default="hw3_data/p2_data/images/val/", help="Image root")
    parser.add_argument("--annotation_file", default="hw3_data/p2_data/val.json", help="Annotation json file")

    args = parser.parse_args()

    evaluate(args)