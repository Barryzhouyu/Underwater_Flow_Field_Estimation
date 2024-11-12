import holoocean
import cv2
import copy
import os.path
import uuid
import os
import numpy as np
from scipy.spatial.transform import Rotation as R  # Import for quaternion conversion
import subprocess

base_cfg = {
    "name": "data collection",
    "world": "SimpleUnderwater",
    "main_agent": "auv0",
    "frames_per_sec": False,
    "ticks_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv1",
            "agent_type": "SphereAgent",
            "sensors": [
                {"sensor_type": "ViewportCapture"},
            ],
            "control_scheme": 0,
            "location": [0, 3, 100],
        },
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "socket": "CameraRightSocket",
                    "configuration": {
                        "CaptureWidth": 256,
                        "CaptureHeight": 256,
                    }
                },
                {
                    "sensor_type": "PoseSensor",
                },
                {
                    "sensor_type": "LocationSensor",
                },
                {
                    "sensor_type": "VelocitySensor",
                },
            ],
            "control_scheme": 0,
            "rotation": [0, 0, 0]
        }
    ],
    "ViewportSize": [1920, 1080]  # Set the desired viewport size (width, height)
}

def sporadic_observation():
    """Makes sure that the RGB camera is positioned and capturing correctly for two configurations."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")
    
    configurations = [
        {"start_location": [0, 4, -4], "velocity": [1.085, -0.074, 0], "T_tot": 3.73, "folder": "captured_image_1"},
        {"start_location": [0, 3, -4], "velocity": [1.158, -0.342, 0], "T_tot": 3.49, "folder": "captured_image_2"},
        {"start_location": [0.5, 7, -4], "velocity": [0.3, -0.841, 0], "T_tot": 3.61, "folder": "captured_image_3"},
        {"start_location": [2.5, 7, -4], "velocity": [0.357, -0.775, 0], "T_tot": 3.90, "folder": "captured_image_4"}
    ]
    
    for index, config in enumerate(configurations):
        # Update the start location in the base config
        base_cfg["agents"][1]["location"] = config["start_location"]
        
        with holoocean.environments.HoloOceanEnvironment(
            scenario=base_cfg,
            binary_path=binary_path,
            show_viewport=True,
            verbose=False,
            uuid=str(uuid.uuid4()),
        ) as env:
            env.spawn_prop("box", [8, -3, -4])
            T_tot = config["T_tot"]
            ticks = int(60 * T_tot)

            velocity_list = []
            location_list = []
            states = env.tick(200)
            
            for i in range(ticks + 1):
                states = env.tick()
                agent = env.agents['auv0']  

                current_location = np.array(states['auv0']["LocationSensor"])  
                current_rotation_matrix = states['auv0']["PoseSensor"][:3, :3]  

                rotation_euler = R.from_matrix(current_rotation_matrix).as_euler('xyz', degrees=False)
                rotation_euler = np.array(rotation_euler)  

                current_location = current_location.astype(np.float32, copy=False)
                rotation_euler = rotation_euler.astype(np.float32, copy=False)

                forward_velocity = np.array(config["velocity"], dtype='float32')  
                angular_velocity = np.array([0, 0, 0], dtype='float32')  
                agent.set_physics_state(location=current_location, rotation=rotation_euler, velocity=forward_velocity, angular_velocity=angular_velocity)

                if i == ticks and i != 0:
                    locations = states['auv0']["LocationSensor"]
                    velocity = states['auv0']["VelocitySensor"]

                    location_list.append(locations)
                    velocity_list.append(velocity)

            list_of_lists = [arr.tolist() for arr in location_list]
            coord_list = [[round(float(item), 2) for item in arr] for arr in list_of_lists]

            # Save captured image in the respective folder
            pixels = states['auv0']['RGBCamera'][:, :, :3]
            directory = os.path.join("/home/zy/Desktop/", config["folder"])
            os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
            xyz = f"{coord_list[-1]}" 
            filepath = os.path.join(directory, f"{xyz}.png")
            cv2.imwrite(filepath, pixels)
            cv2.imshow("Captured Image", pixels)
            cv2.waitKey(1000)
            print(f"Image {index + 1} captured, trajectory {index + 1} finished")
                
    script_path = '/home/zy/Desktop/HoloPy/data_collection/multi_cell_flup.py'
    subprocess.run(['python3', script_path, filepath]) 

    cv2.destroyAllWindows()

if __name__ == "__main__":
    sporadic_observation()