# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "phosphobot",
# ]
#
# ///

from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import ACT
import time
import numpy as np

# Initialize hardware interfaces
client = PhosphoApi(base_url="http://localhost:80")
allcameras = AllCameras()
time.sleep(1)  # Camera warmup

# Connect to ACT server
model = ACT()

while True:
    # Capture multi-camera frames (adjust camera IDs and size as needed)
    images = [
        allcameras.get_rgb_frame(0, resize=(240, 320)),
        allcameras.get_rgb_frame(1, resize=(240, 320)),
        allcameras.get_rgb_frame(2, resize=(240, 320)),
    ]

    # Get current robot state
    state = client.control.read_joints()

    # Generate actions
    actions = model({"state": np.array(state.angles_rad), "images": np.array(images)})

    # Execute actions at 30Hz
    for action in actions:
        client.control.write_joints(angles=action.tolist())
        time.sleep(1 / 30)
