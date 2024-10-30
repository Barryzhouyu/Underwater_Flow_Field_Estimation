import os
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.autograd import grad
import glob
import torchvision.models as models
import torchvision.models as models
from torchvision.models import ResNet50_Weights

################Model#################
class CoordinateToImageResNet(nn.Module):
    def __init__(self, coord_size, image_height, image_width):
        super(CoordinateToImageResNet, self).__init__()

        # Initial transformation
        self.initial_fc = nn.Sequential(
            nn.Linear(coord_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * image_height * image_width),  # Adjust to match the input size
            nn.ReLU(True),
        )

        # Reshape the input to match the image size
        self.image_height = image_height
        self.image_width = image_width

        # Load a pre-trained ResNet model
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust in_channels to match the input
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the last FC layers

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output 3 channels for RGB
        )

        # Final layer to resize to the target image dimensions
        self.final_resize = nn.Upsample(size=(image_height, image_width), mode='bilinear', align_corners=False)

    def forward(self, coords):
        # Transform the input coordinates
        x = self.initial_fc(coords)
        x = x.view(-1, 3, self.image_height, self.image_width)  # Reshape to (B, C, H, W)

        # Pass through the backbone and upsample
        x = self.backbone(x)
        x = self.upsample(x)
        return self.final_resize(x)

model = CoordinateToImageResNet(coord_size=3, image_height=224, image_width=224)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move your model to the appropriate device

# Define loss function and optimizer (if needed)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)