import math
import time
import requests

# Configurations
PI_IP: str = "127.0.0.1"
PI_PORT: int = 80
NUMBER_OF_STEPS: int = 30
NUMBER_OF_CIRCLES: int = 5


# Function to call the API
def call_to_api(endpoint: str, data: dict = {}):
    response = requests.post(f"http://{PI_IP}:{PI_PORT}/move/{endpoint}", json=data)
    return response.json()


# Example code to move the robot in a circle
# 1 - Initialize the robot
call_to_api("init")
print("Initializing robot")
time.sleep(2)

# With the move absolute endpoint, we can move the robot in an absolute position
# 2 - We move the robot in a circle with a diameter of 4 cm
for _ in range(NUMBER_OF_CIRCLES):
    for step in range(NUMBER_OF_STEPS):
        position_y: float = 4 * math.sin(2 * math.pi * step / NUMBER_OF_STEPS)
        position_z: float = 4 * math.cos(2 * math.pi * step / NUMBER_OF_STEPS)
        call_to_api(
            "absolute",
            {
                "x": 0,
                "y": 0,
                "z": position_z,
                "rx": 0,
                "ry": 0,
                # rz is used to move the robot from left to right
                "rz": position_y,
                "open": 0,
            },
        )
        print(f"Step {step} - x: 0, y: {position_y}, z: {position_z}")
        time.sleep(0.03)
