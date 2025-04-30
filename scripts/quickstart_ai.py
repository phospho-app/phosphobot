from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import ACT

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

# Instantiate the model
model = ACT()

# Get the frames from the cameras
# We will use this model: LegrandFrederic/Orange-brick-in-black-box
# It requires 3 cameras as you can see in the config.json
# https://huggingface.co/LegrandFrederic/Orange-brick-in-black-box/blob/main/config.json

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
    ]

    # Get the robot state
    state = client.control.read_joints()

    inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}

    # Go through the model
    actions = model(inputs)

    for action in actions:
        # Send the new joint postion to the robot
        client.control.write_joints(angles=action.tolist())
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
