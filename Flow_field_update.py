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
        self.backbone = models.resnet50(weights=True)
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


################flow field update#################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CoordinateToImageResNet(coord_size=3, image_height=224, image_width=224)
model_path = '/home/zy/Downloads/inverse_best_model_more_epochs_resnet_400.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

alpha = 0.1
T_tot = 5
convergence_threshold = 1e-3
n_consecutive_iterations = 10
speed_of_AUV_x = 0.3
speed_of_AUV_y = 0.05
iterations = 200

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

F_x_old = 0.3
F_y_old = 0.05
convergence_count = 0
Initial_position_AUV = [0, 3, -4]


image_path = glob.glob('/home/zy/Desktop/collected_images/*.png')[0]
#image_path = "/home/zy/Desktop/20_20/['3.56', '-0.00', '-5.00'].png"
print(image_path)

all_updated_F_x = []
all_updated_F_y = []
losses = []

for i in range(iterations):
    actual_image = Image.open(image_path).convert('RGB')
    actual_image_tensor = transform(actual_image).unsqueeze(0).to(device)

    x_new = Initial_position_AUV[0] + (speed_of_AUV_x + F_x_old) * T_tot
    y_new = Initial_position_AUV[1] + (speed_of_AUV_y + F_y_old) * T_tot
    print (x_new, y_new)
    coords = torch.tensor([[x_new, y_new, -4.00]], dtype=torch.float32, device=device, requires_grad=True)

    predicted_image_tensor = model(coords)
    pixel_differences = actual_image_tensor - predicted_image_tensor

    loss = (pixel_differences ** 2).mean()
    loss_value = loss.item()
    losses.append(loss_value)

    loss.backward()

    grad_L_wrt_coords = coords.grad

    F_x_new = F_x_old - alpha * (grad_L_wrt_coords[0][0] * T_tot).item()
    F_y_new = F_y_old - alpha * (grad_L_wrt_coords[0][1] * T_tot).item()

    all_updated_F_x.append(F_x_new)
    all_updated_F_y.append(F_y_new)

    if abs(F_x_new - F_x_old) < convergence_threshold and abs(F_y_new - F_y_old) < convergence_threshold:
        convergence_count += 1
        if convergence_count >= n_consecutive_iterations:
            print("Convergence has been met for several consecutive iterations.")
            break
    else:
        convergence_count = 0

    F_x_old = F_x_new
    F_y_old = F_y_new

print(f"Final flow field values: F_x: {F_x_new:.2f}, F_y: {F_y_new:.2f}")
flow_field_values = {'F_x': all_updated_F_x, 'F_y': all_updated_F_y}
flow_field_filename = '/home/zy/Desktop/HoloPy/data_collection/flow_field_values.pt'
torch.save(flow_field_values, flow_field_filename)
print(f"Flow field values saved to {flow_field_filename}")

updated_flow_field = '/home/zy/Desktop/HoloPy/data_collection/flow_field_values.pt'

flow_field_values = torch.load(updated_flow_field)

F_x_values = flow_field_values['F_x']
F_y_values = flow_field_values['F_y']
#print('F_x_values:', ['{:.2f}'.format(x) for x in F_x_values])
#print('F_y_values:', ['{:.2f}'.format(y) for y in F_y_values])

###################Plot+evaluation######################
F_X_baseline = 0.51
F_Y_baseline = -0.284

plt.figure(figsize=(12, 6))
plt.axhline(y=F_X_baseline, color='blue', linestyle='--', linewidth=2, label='F_X GroundTruth (0.328 m/s)')
plt.axhline(y=F_Y_baseline, color='orange', linestyle='--', linewidth=2, label='F_Y GroundTruth (-0.354 m/s)')
plt.plot(F_x_values, label='Flow in x direction', linewidth=2.0, marker = 'o')
plt.plot(F_y_values, label='Flow in y direction', linewidth=2.0, marker = 'o')
plt.title('Updated Flow Field vs Flow Field GroudTruth')
plt.xlabel('Iterations')
plt.ylabel('Estimated Flow Field (m/s)')
# Add grid
plt.grid(True)
plt.legend()
plt.show()