import numpy as np
import matplotlib.pyplot as plt
import os

# Create a folder to save the plots
output_folder = "/home/zy/Desktop/norm_images"
os.makedirs(output_folder, exist_ok=True)

# Load flow field data for the second plot
actual_flow = np.array([[0.80, 0.15], [0.90, 0.10], [0.90, 0.22], [0.70, 0.20]])

F_x_AUV_4 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_x_4.npy")
F_y_AUV_4 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_y_4.npy")
F_x_AUV_3 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_x_3.npy")
F_y_AUV_3 = np.load("/home/zy/Desktop/HoloPy/data_collection/flow_field_values_y_3.npy")

# Extract flow field data for each cell
F_x_AUV4_cell_1 = F_x_AUV_4[:, 0]
F_y_AUV4_cell_1 = F_y_AUV_4[:, 0]
F_x_AUV4_cell_2 = F_x_AUV_4[:, 1]
F_y_AUV4_cell_2 = F_y_AUV_4[:, 1]

F_x_AUV3_cell_1 = F_x_AUV_3[:, 0]
F_y_AUV3_cell_1 = F_y_AUV_3[:, 0]
F_x_AUV3_cell_2 = F_x_AUV_3[:, 1]
F_y_AUV3_cell_2 = F_y_AUV_3[:, 1]

# Updated flow data for the L2 norm calculation
updated_flow = [
    [F_x_AUV4_cell_1, F_y_AUV4_cell_1], [F_x_AUV4_cell_2, F_y_AUV4_cell_2],
    [F_x_AUV3_cell_1, F_y_AUV3_cell_1], [F_x_AUV3_cell_2, F_y_AUV3_cell_2]
]

# Calculate L2 Norm Differences
iterations = 47
l2_norm_diffs = []

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, iterations)
line, = ax.plot([], [], label="L2 Norm Difference", marker='o', color='b')

ax.set_title("L2 Norm Differences Between Updated and Actual Flow (Iterative)")
ax.set_xlabel("Iteration")
ax.set_ylabel("L2 Norm Difference")
ax.grid(True)
ax.legend()

# Update the plot iteratively and save the images
for i in range(iterations):
    l2_norm_diff_per_iteration = 0
    for cell_index in range(4):
        updated_x = updated_flow[cell_index][0][i]
        updated_y = updated_flow[cell_index][1][i]
        actual_x = actual_flow[cell_index][0]
        actual_y = actual_flow[cell_index][1]
        squared_diff = (updated_x - actual_x) ** 2 + (updated_y - actual_y) ** 2
        l2_norm_diff_per_iteration += squared_diff
    l2_norm_diff = np.sqrt(l2_norm_diff_per_iteration)
    l2_norm_diffs.append(l2_norm_diff)

    # Update the plot
    line.set_data(np.arange(i+1), l2_norm_diffs)  # Set new data for the line
    ax.relim()  # Recompute the data limits based on the new data
    ax.autoscale_view()  # Rescale the y-axis automatically
    plt.pause(0.07)  # Pause for the given time before updating the plot

    # Save each frame as an image
    output_filename = os.path.join(output_folder, f"norm_iteration_{i+1:03d}.png")
    plt.savefig(output_filename)

# Keep the final plot displayed
plt.show()

