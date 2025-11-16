import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AlexNetFCN(nn.Module):
    def __init__(self, classes, dropout_prob: float = 0.2):
        super().__init__()
        alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        for param in alexnet.parameters():
            param.requires_grad = False

        self.features = alexnet.features

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(512, classes, kernel_size=1)
        )

        self.score_pool4 = nn.Conv2d(256, classes, kernel_size=1)
        
        self.upsample_2x = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=2, padding=1)
        self.upsample_16x = nn.ConvTranspose2d(classes, classes, kernel_size=32, stride=16, padding=8)

    def forward(self, x_in):
        input_size = x_in.shape[2:]
        
        pool4 = self.features[:10](x_in)
        pool5 = self.features[10:](pool4)
        
        score = self.classifier(pool5)
        score_pool4 = self.score_pool4(pool4)
        
        score = self.upsample_2x(score)
        
        if score.shape != score_pool4.shape:
            score = F.interpolate(score, size=score_pool4.shape[2:], mode='bilinear', align_corners=False)
        
        score = score + score_pool4
        score = self.upsample_16x(score)
        score = F.interpolate(score, size=input_size, mode='bilinear', align_corners=False)

        return score