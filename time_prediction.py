import numpy as np
import matplotlib.pyplot as plt
import os

# Create a directory to save images
output_folder = "/home/zy/Desktop/time_prediction"
os.makedirs(output_folder, exist_ok=True)

# Load data for travel time
t_min_values_condition_1 = np.load("/home/zy/Desktop/HoloPy/data_collection/t_min_values_condition_1.npy")
t_min_values_condition_2 = np.load("/home/zy/Desktop/HoloPy/data_collection/t_min_values_condition_2.npy")
t_min_values_condition_3 = np.load("/home/zy/Desktop/HoloPy/data_collection/t_min_values_condition_3.npy")
t_min_values_condition_4 = np.load("/home/zy/Desktop/HoloPy/data_collection/t_min_values_condition_4.npy")

# Resample data for other AUVs to match the length of AUV 2
num_iterations = len(t_min_values_condition_2)
iterations_range = np.arange(num_iterations)

def resample_data(data, num_iterations):
    return np.interp(iterations_range, np.linspace(0, num_iterations - 1, len(data)), data)

# Resample travel time values for all AUVs to match AUV 2
t_min_values_condition_1_resampled = resample_data(t_min_values_condition_1, num_iterations)
t_min_values_condition_3_resampled = resample_data(t_min_values_condition_3, num_iterations)
t_min_values_condition_4_resampled = resample_data(t_min_values_condition_4, num_iterations)

# Actual total times for the travel trajectories
T_tot_AUV_1 = 3.49
T_tot_AUV_2 = 3.67
T_tot_AUV_3 = 3.61
T_tot_AUV_4 = 3.90

iterations = num_iterations  

for i in range(iterations):
    # Create the figure
    plt.figure(figsize=(6, 6))

    # Plot predicted travel times
    plt.plot(t_min_values_condition_1_resampled[:i+1], color='g', label='Predicted travel time for T1', linewidth=2.0, marker='o', markersize=5)
    plt.plot(t_min_values_condition_2[:i+1], color='c', label='Predicted travel time for T2', linewidth=2.0, marker='o', markersize=5)
    plt.plot(t_min_values_condition_3_resampled[:i+1], color='gray', label='Predicted travel time for T3', linewidth=2.0, marker='o', markersize=5)
    plt.plot(t_min_values_condition_4_resampled[:i+1], color='orange', label='Predicted travel time for T4', linewidth=2.0, marker='o', markersize=5)

    # Plot actual travel times as horizontal lines
    plt.axhline(y=T_tot_AUV_1, color='g', linestyle='--', linewidth=2, label='Actual total time for T1')
    plt.axhline(y=T_tot_AUV_2, color='c', linestyle='--', linewidth=2, label='Actual total time for T2')
    plt.axhline(y=T_tot_AUV_3, color='gray', linestyle='--', linewidth=2, label='Actual total time for T3')
    plt.axhline(y=T_tot_AUV_4, color='orange', linestyle='--', linewidth=2, label='Actual total time for T4')

    # Add title and labels
    plt.title('Predicted Travel Time for 4 Trajectories vs Actual Travel Time')
    plt.xlabel('Iteration')
    plt.ylabel('Travel time')
    plt.grid(True)
    plt.legend()

    # Save each plot as an image
    output_filename = os.path.join(output_folder, f"iteration_{i + 1:03d}.png")
    plt.savefig(output_filename)

    # Pause to simulate visualization update
    #plt.pause(0.05)
    
    # Close the figure to avoid memory issues
    plt.close()

print("All images saved.")

