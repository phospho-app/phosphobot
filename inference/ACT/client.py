# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "phosphobot",
# ]
#
# ///

from phosphobot.camera import AllCameras
import httpx
from phosphobot.am import ACT
import time
import numpy as np

# Initialize hardware interfaces
PHOSPHOBOT_API_URL = "http://localhost:80"
allcameras = AllCameras()
time.sleep(1)  # Camera warmup

# Connect to ACT server
model = ACT()

while True:
    # Capture multi-camera frames (adjust camera IDs and size as needed)
    images = [allcameras.get_rgb_frame(0, resize=(240, 320))]

    # Get current robot state
    state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()

    # Generate actions
    actions = model({"state": np.array(state["angles"]), "images": np.array(images)})

    # Execute actions at 30Hz
    for action in actions:
        httpx.post(
            f"{PHOSPHOBOT_API_URL}/joints/write", json={"angles": action[0].tolist()}
        )
        time.sleep(1 / 30)
