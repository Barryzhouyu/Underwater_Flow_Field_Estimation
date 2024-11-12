import holoocean
import cv2
import copy
import os.path
import uuid
import os
import numpy as np

# from tests.utils.equality import mean_square_err

base_cfg = {
    "name": "test viewport capture",
    "world": "SimpleUnderwater",
    "main_agent": "auv1",
    "frames_per_sec": False,
    "agents": [
        {
            "agent_name": "auv1",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "socket": "CameraSocket",
                    "configuration": {
                        "CaptureWidth": 1280,
                        "CaptureHeight": 1280,
                    }
                },
                {
                    "sensor_type": "PoseSensor",
                },
                {
                    "sensor_type": "LocationSensor",
                },
            ],
            "control_scheme": 0,
            "location": [6, 0, -4]
        }
    ],
}

def sporadic_observation(auv):
    """Makes sure that the RGB camera is positioned and capturing correctly.

    Capture pixel data, and load from disk the baseline of what it should look like.
    Then, use mse() to see how different the images are.
    """
    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")
    with holoocean.environments.HoloOceanEnvironment(
        scenario=base_cfg,
        binary_path=binary_path,
        show_viewport=True,
        verbose=True,
        uuid=str(uuid.uuid4()),
    ) as env:
        env.spawn_prop("box", [8, 0, -4])
        N = 5
        
        poses = np.zeros((N,4,4))
        # path = "/home/ndroar/Desktop/HoloPy/image_collection/pose.npy"
        if os.path.isfile("/home/roar/Desktop/HoloPy/data_collection/pose.npy"):
            poses_loaded = np.load("/home/roar/Desktop/HoloPy/data_collection/pose.npy")
            loaded = poses_loaded.shape[0]
            poses[:loaded, :, :] = poses_loaded
            print("Loaded", loaded, "saved poses!")
        
        locations = np.zeros((N, 3, 0))
        if os.path.isfile("/home/roar/Desktop/HoloPy/data_collection/locations.npy"):
            locations_loaded = np.load("/home/roar/Desktop/HoloPy/data_collection/locations.npy")
            N_loaded = locations_loaded.shape[0]
            locations[:N_loaded, :, :] = locations_loaded
            print("Loaded", N_loaded, "saved locations!")

        # Camera warm up
        states = env.tick(90)
        n = 0
        location_list = []
        for i in range(500):
            states = env.tick()

            if i == 100*n:
                env.act('auv1', np.array[0,0,0,0,30])
                #pixels = states['auv1']['RGBCamera'][:, :, :3]
                pixels = states['RGBCamera'][:, :, :3]
                #locations=states["auv1"]["LocationSensor"]
                locations=states["LocationSensor"]
                directory="/home/roar/Desktop/testimages/"
                xyz = [f"{x:.2f}" for x in locations]
                filepath = directory + str(xyz) + ".png"
                cv2.imwrite(filepath, pixels)
                #cv2.imshow("image", pixels)
                #cv2.waitKey(0)
                #poses[n, :, :] = states["auv1"]["PoseSensor"]
                poses[n, :, :] = states["PoseSensor"]
                location_list.append(locations)
                n = n + 1

    np.save("/home/roar/Desktop/HoloPy/data_collection/pose.npy", poses)
    np.save("/home/roar/Desktop/HoloPy/data_collection/locations.npy", location_list)
cv2.destroyAllWindows()
if __name__ == "__main__":
    sporadic_observation("auv")