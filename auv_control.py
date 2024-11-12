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
}

def sporadic_observation():
    """Makes sure that the RGB camera is positioned and capturing correctly for all configurations without reopening the simulator."""
    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")
    
    configurations = [
        {"start_location": [0, 4, -4.00], "velocity": [0.839, 1.939, 0], "T_tot": 3.73, "folder": "captured_image_3"},
        {"start_location": [0, 4, -4.00], "velocity": [0.939, 0.939, 0], "T_tot": 3.73, "folder": "captured_image_3"},
        {"start_location": [0, 4, -4.00], "velocity": [1.057, 0.472, 0], "T_tot": 3.73, "folder": "captured_image_1"},
        {"start_location": [0, 4, -4.00], "velocity": [1.155, 0.342, 0], "T_tot": 3.73, "folder": "captured_image_2"},
        {"start_location": [0, 4, -4.00], "velocity": [1.089, 0.153, 0], "T_tot": 3.73, "folder": "captured_image_4"},
        {"start_location": [0, 4, -4.00], "velocity": [1.085, -0.074, 0], "T_tot": 3.73, "folder": "captured_image_4"},
    ]

    with holoocean.environments.HoloOceanEnvironment(
        scenario=base_cfg,
        binary_path=binary_path,
        show_viewport=True,  # Disable viewport
        verbose=False,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.spawn_prop("box", [8, -3, -4])
        print("AUV Sink")
        print("Warming up the onboard camera...")
        for _ in range(200): 
            env.tick()
        print("Start Navigation")

        for index, config in enumerate(configurations):
            env.agents['auv0'].teleport(config["start_location"]) 
            T_tot = config["T_tot"]
            ticks = int(60 * T_tot)

            velocity_list = []
            location_list = []

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
            cv2.waitKey(100)
            
            # for velocity in velocity_list:
            #     formatted_velocity = [f"{v:.2f}" for v in velocity]
            #     print(f"Velocity: {formatted_velocity}")
                
            for location in location_list:
                formatted_location = [f"{r:.2f}" for r in location]
                print(f"location: {formatted_location}")            

            # Print that image is captured and the trajectory is finished
            #print(f"Image {index + 1} captured, trajectory {index + 1} finished")
            
    cv2.destroyAllWindows()
    print(f"Initiating Flow Field Update Procedure...")
    # script_path = '/home/zy/Desktop/HoloPy/data_collection/multi_cell_flup.py'
    # subprocess.run(['python3', script_path, filepath]) 

if __name__ == "__main__":
    sporadic_observation()



                
        #script_path = '/home/zy/Desktop/HoloPy/data_collection/Flow_field_update.py'
        #subprocess.run(['python3', script_path, filepath]) 

