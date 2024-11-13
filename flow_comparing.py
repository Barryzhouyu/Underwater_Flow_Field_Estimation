import numpy as np
import matplotlib.pyplot as plt
import os
import time
import subprocess

# Actual flow data (inverted X and Y)
actual_flow = np.array([
    [0.15, 0.80],  
    [0.10, 0.90],  
    [0.22, 0.90],  
    [0.20, 0.70]   
])

# Load saved flow field values
F_x_AUV_4 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_x_4.npy")
F_y_AUV_4 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_y_4.npy")
F_x_AUV_3 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_x_3.npy")
F_y_AUV_3 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_y_3.npy")

# Invert X and Y for cells
F_x_AUV4_cell_1 = F_y_AUV_4[:, 0]  
F_y_AUV4_cell_1 = F_x_AUV_4[:, 0]
F_x_AUV4_cell_2 = F_y_AUV_4[:, 1]  
F_y_AUV4_cell_2 = F_x_AUV_4[:, 1]
F_x_AUV3_cell_1 = F_y_AUV_3[:, 0]  
F_y_AUV3_cell_1 = F_x_AUV_3[:, 0]
F_x_AUV3_cell_2 = F_y_AUV_3[:, 1]  
F_y_AUV3_cell_2 = F_x_AUV_3[:, 1]

updated_flow = [
    [F_x_AUV4_cell_1, F_y_AUV4_cell_1],  
    [F_x_AUV4_cell_2, F_y_AUV4_cell_2],  
    [F_x_AUV3_cell_1, F_y_AUV3_cell_1],  
    [F_x_AUV3_cell_2, F_y_AUV3_cell_2]   
]

# Create a folder to save the plots (or for video frames)
output_folder = "/home/zy/Desktop/flow_comparing_imageforv"
os.makedirs(output_folder, exist_ok=True)

# Iterations count
iterations = F_x_AUV_4.shape[0]

output_folder = "/home/zy/Desktop/flow_comparing_images"
os.makedirs(output_folder, exist_ok=True)

# Limit to first 47 iterations or less if fewer iterations are available
max_iterations = min(iterations, 47)

# Plot dimensions (for each subplot position in the 2x2 grid)
cell_positions = [
    [1, 2],  # Cell 1
    [2, 2],  # Cell 2
    [1, 1],  # Cell 3
    [2, 1]   # Cell 4
]

fig, ax = plt.subplots(2, 2, figsize=(6, 6))

for axes in ax.flatten():
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_aspect('equal')
    axes.set_xticks([])  # Remove x-axis ticks
    axes.set_yticks([])  # Remove y-axis ticks
    
for i in range(2):  # Loop over rows
    for j in range(2):  # Loop over columns
        # Plot vertical and horizontal boundaries in each subplot
        ax[i, j].plot([-1, 1], [0, 0], color='black', linewidth=2)  # Horizontal boundary
        ax[i, j].plot([0, 0], [-1, 1], color='black', linewidth=2)  # Vertical boundary

# Create a plot for each iteration (limit to first 47 iterations)
for iteration in range(max_iterations):

    # Iterate over each cell to plot actual and updated flows
    for cell_index, (updated_x, updated_y) in enumerate(updated_flow):
        actual_x = actual_flow[cell_index][0]
        actual_y = actual_flow[cell_index][1]

        row, col = cell_positions[cell_index]

        # Clear the previous plot in the cell
        ax[row-1, col-1].cla()

        # Set limits and aspect ratio (again, to avoid changes during updates)
        ax[row-1, col-1].set_xlim(-1, 1)
        ax[row-1, col-1].set_ylim(-1, 1)
        ax[row-1, col-1].set_aspect('equal')
        ax[row-1, col-1].set_xticks([])  # Remove x-axis ticks
        ax[row-1, col-1].set_yticks([])  # Remove y-axis ticks

        # Actual flow in blue
        ax[row-1, col-1].quiver(0, 0, actual_x, actual_y, color='b', angles='xy', scale_units='xy', scale=1, width=0.01)
        # Updated flow in orange
        ax[row-1, col-1].quiver(0, 0, updated_x[iteration], updated_y[iteration], color='orange', angles='xy', scale_units='xy', scale=1, width=0.01)

        # Set the title inside the cell (e.g., 1, 2, 3, 4)
        ax[row-1, col-1].text(0.5, 0.5, str(cell_index + 1), fontsize=12, ha='center')

    # Overall plot title
    fig.suptitle(f"Iteration {iteration + 1}: Comparison of Actual and Predicted Flow Fields", fontsize=14)

    # Save each frame as an image
    plt.tight_layout()
    output_filename = os.path.join(output_folder, f"iteration_{iteration + 1:03d}.png")
    plt.savefig(output_filename)

    plt.pause(0.01)  # Adjust pause duration for visualization speed

time.sleep(3)

plt.close(fig)

print("Results of Time Prediction and Trajectory Tracing will display...")

script_path = '/home/zy/Desktop/HoloPy/data_collection/flow_plots.py'
subprocess.run(['python3', script_path]) 





