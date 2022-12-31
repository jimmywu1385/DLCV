import torch.nn as nn
from transformers import WavLMForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torch.nn.functional as F
import torchvision.models as models

class waveLM_Resnet152(nn.Module):
    def __init__(self, num_label=2, hidden_size=128, audio_backbone="microsoft/wavlm-base-plus"):
        super(waveLM_Resnet152, self).__init__()
        self.video_encoder = Resnet152Encoder()

        self.audio_encoder = WavLMForSequenceClassification.from_pretrained(audio_backbone, num_labels=2)
        self.audio_projector = nn.Linear(self.audio_encoder.config.hidden_size, hidden_size)

        self.max_duration = 8.0

    def forward(self, audio_inputs, video_inputs, video_length):
        audio = self.audio_encoder(**audio_inputs).logits

        video = self.video_encoder(video_inputs)
        # print("audio", audio.shape)
        # print("video", video.shape)

        return 0.7 * audio + 0.3 * video


class Resnet152Encoder(nn.Module):
    def __init__(self, num_class=2):
        super(Resnet152Encoder, self).__init__()
        
        self.num_class = num_class
        self.backbone = models.resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        self.fc = nn.Linear(10240, num_class)

    def forward(self, x):
        # dim: batch_size * 512 * 1 * 1
        x1 = self.feature_extractor(x[:, 0])
        x2 = self.feature_extractor(x[:, 1])
        x3 = self.feature_extractor(x[:, 2])
        x4 = self.feature_extractor(x[:, 3])
        x5 = self.feature_extractor(x[:, 4])
        
        X = torch.cat((x1, x2, x3, x4, x5), 1)
        X = torch.squeeze(X, 2)
        X = torch.squeeze(X, 2)
        # dim: batch_size * (512*5)
        y = self.fc(X)
        return y

if __name__ == "__main__":
    images = torch.randn((1, 16, 3, 224, 224))
    audio = torch.randn((1, 30720, 1))
