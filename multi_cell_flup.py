import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import numpy as np
from CNN_model import CoordinateToImageResNet
from cell_time import calculate_navigation_time
from PIL import Image
import glob
import matplotlib.pyplot as plt
import subprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CoordinateToImageResNet(coord_size=3, image_height=224, image_width=224)
model_path = '/home/zy/Desktop/HoloPy/data_collection/inverse_best_model_more_epochs_resnet.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

convergence_threshold = 1e-3
n_consecutive_iterations = 39
iterations = 200
learning_rate = 0.1
cell_length_x = 2
cell_length_y = 2
convergence_count = 0
lam = 1
epsilon = 1e-5
lambda_reg = 1
grad_J2_x = np.zeros((2, 1))
grad_J2_y = np.zeros((2, 1))

F_x_old_bottom_1 = np.array([0.8, 0.65]).reshape(2, 1)
F_y_old_bottom_1 = np.array([0.15, 0.12]).reshape(2, 1)

F_x_old_bottom_2 = np.array([0.55, 0.7]).reshape(2, 1)
F_y_old_bottom_2 = np.array([0.13, 0.1]).reshape(2, 1)

F_x_old_left_1 = np.array([0.8, 0.55]).reshape(2, 1)
F_y_old_left_1 = np.array([0.15, 0.13]).reshape(2, 1)

F_x_old_left_2 = np.array([0.65, 0.7]).reshape(2, 1)
F_y_old_left_2 = np.array([0.12, 0.1]).reshape(2, 1)

# F_x_old_bottom_1 = np.array([0.4, 0.325]).reshape(2, 1)
# F_y_old_bottom_1 = np.array([0.075, 0.06]).reshape(2, 1)

# F_x_old_bottom_2 = np.array([0.275, 0.35]).reshape(2, 1)
# F_y_old_bottom_2 = np.array([0.065, 0.05]).reshape(2, 1)

# F_x_old_left_1 = np.array([0.4, 0.275]).reshape(2, 1)
# F_y_old_left_1 = np.array([0.075, 0.065]).reshape(2, 1)

# F_x_old_left_2 = np.array([0.325, 0.35]).reshape(2, 1)
# F_y_old_left_2 = np.array([0.06, 0.05]).reshape(2, 1)


conditions = [
    {
        'v_x': np.array([0.3, 0.3]).reshape(2, 1),
        'v_y': np.array([0.05, 0.05]).reshape(2, 1),
        'Initial_position_AUV': [0, 1, -4],
        'image_path': "/home/zy/Desktop/captured_image_2/[4.0, 1.82, -4.0].png",
        'F_x_old': F_x_old_bottom_1,
        'F_y_old': F_y_old_bottom_1,
        'T_tot': 3.49
    },
    {
        'v_x': np.array([0.3, 0.3]).reshape(2, 1),
        'v_y': np.array([0.05, 0.05]).reshape(2, 1),
        'Initial_position_AUV': [0, 3, -4],
        'image_path': "/home/zy/Desktop/captured_image_1/[4.0, 3.73, -4.0].png",
        'F_x_old': F_x_old_bottom_2,
        'F_y_old': F_y_old_bottom_2,
        'T_tot': 3.67
    },
    {
        'v_x': np.array([-0.5, -0.5]).reshape(2, 1),
        'v_y': np.array([0.9, 0.9]).reshape(2, 1),
        'Initial_position_AUV': [0.5, 0, -4],
        'image_path': "/home/zy/Desktop/captured_image_3/[1.57, 4.0, -4.0].png",
        'F_x_old': F_x_old_left_1,
        'F_y_old': F_y_old_left_1,
        'T_tot': 3.61
    },
    {
        'v_x': np.array([-0.5, -0.5]).reshape(2, 1),
        'v_y': np.array([0.9, 0.9]).reshape(2, 1),
        'Initial_position_AUV': [2.5, 0, -4],
        'image_path': "/home/zy/Desktop/captured_image_4/[3.88, 4.0, -4.0].png",
        'F_x_old': F_x_old_left_2,
        'F_y_old': F_y_old_left_2,
        'T_tot': 3.90
    }
    
    # {
    #     'v_x': np.array([-0.25, -0.25]).reshape(2, 1),
    #     'v_y': np.array([0.9, 0.9]).reshape(2, 1),
    #     'Initial_position_AUV': [0.5, 0, -4],
    #     'image_path': "/home/zy/Desktop/captured_image_3/[1.57, 4.0, -4.0].png",
    #     'F_x_old': F_x_old_left_1,
    #     'F_y_old': F_y_old_left_1,
    #     'T_tot': 3.61
    # },
    # {
    #     'v_x': np.array([-0.25, -0.25]).reshape(2, 1),
    #     'v_y': np.array([0.9, 0.9]).reshape(2, 1),
    #     'Initial_position_AUV': [2.5, 0, -4],
    #     'image_path': "/home/zy/Desktop/captured_image_4/[3.88, 4.0, -4.0].png",
    #     'F_x_old': F_x_old_left_2,
    #     'F_y_old': F_y_old_left_2,
    #     'T_tot': 3.90
    # }
]

final_position_cell_one = []
final_position_cell_two = []

for idx, condition in enumerate(conditions):
    v_x = condition['v_x']
    v_y = condition['v_y']
    Initial_position_AUV = condition['Initial_position_AUV']
    image_path = condition['image_path']
    F_x_old = condition['F_x_old']
    F_y_old = condition['F_y_old']
    T_tot = condition['T_tot'] 

    t_min_values = []
    losses = []
    all_updated_F_x = [F_x_old.copy()]
    all_updated_F_y = [F_y_old.copy()]

    # Initialize the lists to store final positions for this condition
    final_position_cell_one = []
    final_position_cell_two = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(iterations):
        actual_image = Image.open(image_path).convert('RGB')
        actual_image_tensor = transform(actual_image).unsqueeze(0).to(device)

        # Calculate the time and positions for cell 1
        t_min_1, t_xy_1 = calculate_navigation_time(
            Initial_position_AUV, cell_length_x, cell_length_y, v_x[0], v_y[0], F_x_old[0], F_y_old[0]
        )
        x_new_1 = Initial_position_AUV[0] + (v_x[0] + F_x_old[0]) * t_min_1
        y_new_1 = Initial_position_AUV[1] + (v_y[0] + F_y_old[0]) * t_min_1

        # Store scalar values
        x_new_1_scalar = x_new_1.item()
        y_new_1_scalar = y_new_1.item()

        # Handling small position deviations
        if abs(x_new_1_scalar - 2.0) < epsilon:
            x_new_1_scalar = 2.0
        if abs(y_new_1_scalar - 2.0) < epsilon:
            y_new_1_scalar = 2.0

        # Calculate the time and positions for cell 2
        t_min_2, t_xy_2 = calculate_navigation_time(
            [x_new_1_scalar, y_new_1_scalar, -4], cell_length_x, cell_length_y, v_x[1], v_y[1], F_x_old[1], F_y_old[1]
        )
        x_new_2 = x_new_1_scalar + (v_x[1] + F_x_old[1]) * t_min_2
        y_new_2 = y_new_1_scalar + (v_y[1] + F_y_old[1]) * t_min_2
        
        x_new_2_scalar = x_new_2.item()
        y_new_2_scalar = y_new_2.item()

        # Save final positions in both cells
        final_position_cell_one.append([x_new_1_scalar, y_new_1_scalar])
        final_position_cell_two.append([x_new_2_scalar, y_new_2_scalar])

        # Process for flow field updating
        t_tot = np.array([t_min_1, t_min_2]).reshape(2, 1)
        final_positions = np.array([[x_new_2], [y_new_2]])

        coords = torch.tensor([[x_new_2_scalar, y_new_2_scalar, -4.00]], dtype=torch.float32, device=device, requires_grad=True)
        predicted_image_tensor = model(coords)
        pixel_differences = actual_image_tensor - predicted_image_tensor

        # Loss calculations
        t_total = np.sum(t_tot)
        t_min_values.append(t_total)

        time_differences = T_tot - t_total

        # Primary loss (J1)
        J1 = (pixel_differences ** 2).mean()
        losses.append(J1.item())

        # Regularization loss (J2)
        J2 = lambda_reg * (torch.sum(torch.tensor(F_x_old**2)) + torch.sum(torch.tensor(F_y_old**2)))

        # Total loss
        total_loss = J1 + J2
        total_loss.backward()

        # Gradients
        grad_L_wrt_coords = coords.grad
        grad_J1_x = grad_L_wrt_coords[0][0].cpu().numpy() * t_tot
        grad_J1_y = grad_L_wrt_coords[0][1].cpu().numpy() * t_tot

        # Calculate gradients for both cells
        for j in range(2):
            if j == 0:
                if t_min_1 == t_xy_1[0]:
                    grad_J2_x[j] = lam * time_differences * (t_min_1 / (v_x[j] + F_x_old[j]))
                    grad_J2_y[j] = 0
                else:
                    grad_J2_y[j] = lam * time_differences * (t_min_1 / (v_y[j] + F_y_old[j]))
                    grad_J2_x[j] = 0
            elif j == 1:
                if t_min_2 == t_xy_2[0]:
                    grad_J2_x[j] = lam * time_differences * (t_min_2 / (v_x[j] + F_x_old[j]))
                    grad_J2_y[j] = 0
                else:
                    grad_J2_y[j] = lam * time_differences * (t_min_2 / (v_y[j] + F_y_old[j]))
                    grad_J2_x[j] = 0

        grad_G_x = grad_J1_x + grad_J2_x
        grad_G_y = grad_J1_y + grad_J2_y

        # Update flow fields
        F_x_new = F_x_old - learning_rate * grad_G_x
        F_y_new = F_y_old - learning_rate * grad_G_y

        all_updated_F_x.append(F_x_new.copy())
        all_updated_F_y.append(F_y_new.copy())

        # Convergence check
        if np.all(abs(F_x_new - F_x_old) < convergence_threshold) and np.all(abs(F_y_new - F_y_old) < convergence_threshold):
            convergence_count += 1
            if convergence_count >= n_consecutive_iterations:
                print("Convergence has been met for several consecutive iterations.")
                break
        else:
            convergence_count = 0

        F_x_old = F_x_new
        F_y_old = F_y_new

        print(f"Updated flow: F_x: {F_x_new}, F_y: {F_y_new}")
        print("Final positions:", final_positions)

    # Save all necessary data at the end of iterations
    np.save(f"/home/zy/Desktop/HoloPy/data_collection/flow_field_values_x_{idx+1}.npy", np.array(all_updated_F_x))
    np.save(f"/home/zy/Desktop/HoloPy/data_collection/flow_field_values_y_{idx+1}.npy", np.array(all_updated_F_y))
    np.save(f"/home/zy/Desktop/HoloPy/data_collection/t_min_values_condition_{idx+1}.npy", np.array(t_min_values))

    # Save final positions for both cells
    np.save(f"/home/zy/Desktop/HoloPy/data_collection/final_position_cell_one_{idx+1}.npy", np.array(final_position_cell_one))
    np.save(f"/home/zy/Desktop/HoloPy/data_collection/final_position_cell_two_{idx+1}.npy", np.array(final_position_cell_two))

    print(f"Final position in cell one for condition {idx+1} saved: [{x_new_1_scalar}, {y_new_1_scalar}]")
    print(f"Final position in cell two for condition {idx+1} saved: [{x_new_2_scalar}, {y_new_2_scalar}]")


    
script_path_1 = '/home/zy/Desktop/HoloPy/data_collection/flow_comparing.py'
script_path_2 = '/home/zy/Desktop/HoloPy/data_collection/norm_iter.py'

process_1 = subprocess.Popen(['python3', script_path_1])
process_2 = subprocess.Popen(['python3', script_path_2])

