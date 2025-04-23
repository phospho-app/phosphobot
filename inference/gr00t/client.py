# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cv2",
#     "phosphobot",
#     "torch",
#     "zmq",
# ]
# ///
import time

import cv2
import numpy as np

from phosphobot.am import Gr00tN1
from phosphobot.api.client import PhosphoApi
from phosphobot.camera import AllCameras

host = "YOUR_SERVER_IP"  # Change this to your server IP (this is the IP of the machine running the Gr00tN1 server using a GPU)
port = 5555

# Change this with your task description
TASK_DESCRIPTION = (
    "Pick up the green lego brick from the table and put it in the black container."
)

# Connect to the phosphobot server, this is different from the server IP above
client = PhosphoApi(base_url="http://localhost:80")

allcameras = AllCameras()
time.sleep(1)  # Wait for the cameras to initialize

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(320, 240)),
        allcameras.get_rgb_frame(camera_id=1, resize=(320, 240)),
    ]

    for i in range(0, len(images)):
        image = images[i]
        if image is None:
            print(f"Camera {i} is not available.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add a batch dimension (from (240, 320, 3) to (1, 240, 320, 3))
        converted_array = np.expand_dims(image, axis=0)
        converted_array = converted_array.astype(np.uint8)
        images[i] = converted_array

    # Create the model
    model = Gr00tN1(server_url=host, server_port=port)

    state = np.array(client.control.read_joints().angles_rad)
    # Take a look at the experiment_cfg/metadata.json file in your Gr00t model and check the names of the images, states, and observations
    # You may need to adapt the obs JSON to match these names
    # The following JSON should work for one arm and 2 video cameras
    obs = {
        "video.image_cam_0": images[0],
        "video.image_cam_1": images[1],
        "state.arm": state[0:6].reshape(1, 6),
        "annotation.human.action.task_description": [TASK_DESCRIPTION],
    }
    # Uncomment this if you want to use this script for a bimanual setup
    #
    # You may also need to change the name and order of the images based on their incoming order and names in the experiment_cfg/metadata.json file in your model
    #
    # obs = {
    #     "video.image_cam_0": images[0],
    #     "video.image_cam_1": images[1],
    #     "video.image_cam_2": images[2],
    #     "state.arm_0": state[0:6].reshape(1, 6),
    #     "state.arm_1": state[6:12].reshape(1, 6),
    #     "annotation.human.action.task_description": [TASK_DESCRIPTION],
    # }

    action = model.sample_actions(obs)

    for i in range(0, action.shape[0]):
        client.control.write_joints(angles=action[i])
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
