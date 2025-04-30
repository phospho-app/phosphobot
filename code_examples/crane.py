#!/usr/bin/env python3

"""
This script let's you control your robot using keyboard inputs.
The robot arm is controlled using the following keys:
- Arrow Up:    Move forward (increase Y)
- Arrow Down:  Move backward (decrease Y)
- Arrow Right: Move up (increase Z)
- Arrow Left:  Move down (decrease Z)
- A:           Move left (decrease X)
- D:           Move right (increase X)
- Space:       Toggle open state
"""

import requests
import time
import logging
from typing import cast
from pynput.keyboard import KeyCode
from typing import Dict, Literal, Set, Tuple
from pynput import keyboard as pynput_keyboard


# Configuration
BASE_URL: str = "http://127.0.0.1:80/"
STEP_SIZE: int = 2  # Movement step in centimeters
SLEEP_TIME: float = 0.05  # Loop sleep time (20 Hz)

# Global open state (initially 1 as set in init_robot)
open_state: Literal[0, 1] = 1


def behind_or_front() -> Literal["Behind", "Facing"]:
    """
    Returns the relative position of the user to the robot.
    """
    while True:
        inp = input(
            "Type 'Behind' if you are behind your robot or 'Facing' if you are facing your robot: "
        )
        if inp in ["Behind", "Facing"]:
            print(f"You chose '{inp}'")
            inp = cast(Literal["Behind", "Facing"], inp)
            return inp
        else:
            print("You must choose between 'Behind' or 'Facing'")


user_position: Literal["Behind", "Facing"] = behind_or_front()

# Key mappings for alphanumeric keys (x-axis movement)
KEY_MAPPINGS: Dict[str, Tuple[int, int, int]] = {
    "a": (0, 0, STEP_SIZE),  # Increase Z (move up)
    "d": (0, 0, -STEP_SIZE),  # Decrease 2 (move down)
}

# Key mappings for special keys (arrow keys for y and z axes)
# The direction of movement is reversed if the user is facing the robot.
SPECIAL_KEY_MAPPINGS: Dict[KeyCode | str, Tuple[int, int, int]] = (
    {
        pynput_keyboard.Key.up: (STEP_SIZE, 0, 0),  # Increase X (move forward)
        pynput_keyboard.Key.down: (-STEP_SIZE, 0, 0),  # Decrease X (move backward)
        pynput_keyboard.Key.right: (0, -STEP_SIZE, 0),  # Increase Y (move right)
        pynput_keyboard.Key.left: (0, STEP_SIZE, 0),  # Decrease Y (move left)
    }
    if user_position == "Behind"
    else {
        pynput_keyboard.Key.up: (-STEP_SIZE, 0, 0),  # Increase X (move forward)
        pynput_keyboard.Key.down: (STEP_SIZE, 0, 0),  # Decrease X (move backward)
        pynput_keyboard.Key.right: (0, STEP_SIZE, 0),  # Increase Y (move right)
        pynput_keyboard.Key.left: (0, -STEP_SIZE, 0),  # Decrease Y (move left)
    }
)

# Set to track currently pressed keys (both string and special keys)
keys_pressed: Set[KeyCode | str] = set()


def toggle_open_state() -> None:
    """Toggle the open state and send a relative move command with no displacement."""
    global open_state
    open_state = 0 if open_state == 1 else 1
    data: Dict[str, float] = {
        "x": 0,
        "y": 0,
        "z": 0,
        "rx": 0,
        "ry": 0,
        "rz": 0,
        "open": open_state,
    }
    try:
        response = requests.post(f"{BASE_URL}move/relative", json=data, timeout=1)
        response.raise_for_status()
        logging.info(f"Toggled open state to {open_state}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to toggle open state: {e}")


def on_press(key: KeyCode | str) -> None:
    # Handle space key separately to toggle open staaaate immediately.
    if key == pynput_keyboard.Key.space:
        toggle_open_state()
        return
    # Handle alphanumeric keys (e.g. "a", "d")
    try:
        char = key.char.lower()  # type: ignore
        if char in KEY_MAPPINGS:
            keys_pressed.add(char)
    except AttributeError:
        # Handle special keys (e.g. arrow keys)
        if key in SPECIAL_KEY_MAPPINGS:
            keys_pressed.add(key)


def on_release(key: KeyCode | str) -> None:
    # Remove keys from the pressed set when released.
    try:
        char = key.char.lower()  # type: ignore
        keys_pressed.discard(char)
    except AttributeError:
        if key in SPECIAL_KEY_MAPPINGS:
            keys_pressed.discard(key)


def init_robot() -> None:
    """Initialize the robot by calling /move/init and setting an absolute starting position."""
    endpoint_init = f"{BASE_URL}move/init"
    endpoint_absolute = f"{BASE_URL}move/absolute"
    try:
        response = requests.post(endpoint_init, json={}, timeout=5)
        response.raise_for_status()
        time.sleep(2)
        response = requests.post(
            endpoint_absolute,
            json={"x": 0, "y": 0, "z": 0, "rx": 1.5, "ry": 0, "rz": 0, "open": 1},
            timeout=5,
        )
        response.raise_for_status()
        logging.info("Robot initialized successfully")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to initialize robot: {e}")
    time.sleep(1)  # Allow time for the robot to initialize


def control_robot():
    """Control the robot with keyboard inputs using /move/relative."""
    endpoint = f"{BASE_URL}move/relative"

    logging.info("Control the end effector using the following keys:")
    logging.info("  Arrow Up:    Move forward (increase RZ)")
    logging.info("  Arrow Down:  Move backward (decrease RZ)")
    logging.info("  Arrow Right: Move up (increase Z)")
    logging.info("  Arrow Left:  Move down (decrease Z)")
    logging.info("  A:           Move left (decrease X)")
    logging.info("  D:           Move right (increase X)")
    logging.info("  Space:       Toggle open state")
    logging.info("Press Ctrl+C to exit")

    # Start the pynput keyboard listener.
    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            # Reset movement deltas.
            delta_x, delta_rz, delta_z = 0.0, 0.0, 0.0

            # Aggregate movement contributions from pressed keys.
            for key in keys_pressed:
                if isinstance(key, str):  # Alphanumeric keys.
                    dx, drz, dz = KEY_MAPPINGS.get(key, (0, 0, 0))
                else:  # Special keys (arrow keys).
                    dx, drz, dz = SPECIAL_KEY_MAPPINGS.get(key, (0, 0, 0))
                delta_x += dx
                delta_rz += drz
                delta_z += dz

            global open_state
            # Send a relative move command if any movement key is pressed.
            if delta_x or delta_rz or delta_z:
                data = {
                    "x": delta_x,
                    "y": 0,
                    "z": delta_z,
                    "rx": 0,
                    "ry": 0,
                    "rz": delta_rz,
                    "open": open_state,
                }
                try:
                    response = requests.post(endpoint, json=data, timeout=1)
                    response.raise_for_status()
                    logging.info(
                        f"Sent movement: x={delta_x}, y={delta_rz}, z={delta_z}"
                    )
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request failed: {e}")

            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        logging.info("Exiting...")
    finally:
        listener.stop()


def main():
    init_robot()
    control_robot()


if __name__ == "__main__":
    main()
