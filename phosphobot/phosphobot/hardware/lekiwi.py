import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import zmq
from loguru import logger

from phosphobot.hardware.base import BaseMobileRobot
from phosphobot.models import RobotConfigStatus


@dataclass
class MovementCommand:
    position: np.ndarray
    orientation: Optional[np.ndarray] = None


class LeKiwi(BaseMobileRobot):
    name = "lekiwi"

    def __init__(self, ip: str, port: int, max_history_len: int = 100, **kwargs):
        """
        Initialize the LeKiwi robot.

        Args:
            ip: IP address of the robot
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.ip = ip
        self.port = port
        self.conn = None
        self.current_position = np.zeros(3)  # [x, y, z]
        self.current_orientation = np.zeros(3)  # [roll, pitch, yaw]
        self._is_connected = False

        # Track movement instructions
        self.movement_queue: Deque[MovementCommand] = deque(maxlen=max_history_len)

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected and self.conn is not None

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """
        Set the connection status of the robot.

        Args:
            value: True if connected, False otherwise
        """
        if value and self.conn is None:
            raise ValueError("Cannot set is_connected to True without a connection")
        self._is_connected = value

    async def connect(self) -> None:
        """
        Initialize communication with the robot.
        This method creates a zmq context and connects to the robot's data channel.

        Raises:
            Exception: If the connection fails
        """
        try:
            self.context = zmq.Context()
            self.cmd_socket = self.context.socket(zmq.PUSH)
            connection_string = f"tcp://{self.ip}:{self.port}"
            self.cmd_socket.connect(connection_string)
            self.cmd_socket.setsockopt(zmq.CONFLATE, 1)

            self.is_connected = True
            logger.info("Successfully connected to LeKiwi")

        except Exception as e:
            logger.warning(f"Failed to connect to LeKiwi: {e}")
            raise Exception(f"Failed to connect to LeKiwi: {e}")

    def disconnect(self) -> None:
        """
        Close the connection to the robot.
        """
        if self.conn:
            self.cmd_socket.close()
            self.context.term()

            self.conn = None
            self.is_connected = False

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.

        Returns:
            - state: Last movement command position [x, y, z] and orientation [roll, pitch, yaw]
            - joints_position: Empty array as we're not tracking joint positions
        """
        # Return the last movement command as observation
        # Combine position and orientation into a 6D state vector
        state = np.concatenate([self.current_position, self.current_orientation])
        # Return empty array for joints since we're not tracking them
        joints_position = np.array([])

        return state, joints_position

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Not implemented
        """
        pass

    def get_info_for_dataset(self):
        """
        Not implemented
        """
        raise NotImplementedError

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,
        target_orientation_rad: Optional[np.ndarray],
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation asynchronously.

        Args:
            target_position: Target position as [x, y, z]
                x = forward/backward
                y = right/left
                z = rotate right/ rotate left
            target_orientation_rad: Target orientation in radians [roll, pitch, yaw]
                ignored for this robot due to physics
            **kwargs: Additional arguments
        """
        if not self.is_connected or self.conn is None:
            logger.warning("Robot is not connected")
            return

        x_cmd = target_position[0]  # forward/backward
        y_cmd = target_position[1]  # right/left
        theta_cmd = target_position[2]  # rotate right/rotate left

        wheel_commands = self.body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

        self.movement_queue.append(
            MovementCommand(
                position=target_position.copy(),
                orientation=target_orientation_rad.copy()
                if target_orientation_rad is not None
                else None,
            )
        )

        self.current_position = target_position
        if target_orientation_rad is not None:
            self.current_orientation = target_orientation_rad

        # Send the command to the robot
        message = {"raw_velocity": wheel_commands, "arm_positions": []}
        self.cmd_socket.send_string(json.dumps(message))

        logger.info(
            f"Moved to position {target_position} with orientation {target_orientation_rad}"
        )

    @classmethod
    def from_ip(cls, ip: str, port: int, **kwargs) -> Optional["LeKiwi"]:
        """
        Create an instance from an IP and port

        Args:
            ip: IP address of the robot
            **kwargs: Additional arguments

        Returns:
            LeKiwi instance or None if the connection fails
        """
        try:
            robot = cls(ip=ip, port=port, **kwargs)
            return robot
        except Exception as e:
            logging.error(f"Failed to connect to LeKiwi at {ip}:{port} {e}")
            return None

    def status(self) -> RobotConfigStatus:
        """
        Get the status of the robot.

        Returns:
            RobotConfigStatus object
        """
        return RobotConfigStatus(
            name=self.name, device_name=self.ip, robot_type="mobile"
        )

    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.
        This puts the robot in a stand position ready for operation.
        """
        pass

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.

        This makes the robot sit down before potentially disconnecting.
        """
        if not self.is_connected or self.conn is None:
            logger.warning("Robot is not connected")
            return

    def body_to_wheel_raw(
        self,
        x_cmd: float,
        y_cmd: float,
        theta_cmd: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"left_wheel": value, "back_wheel": value, "right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta_cmd * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Define the wheel mounting angles (defined from y axis cw)
        angles = np.radians(np.array([300, 180, 60]))
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [LeKiwi.degps_to_raw(deg) for deg in wheel_degps]

        return {
            "left_wheel": wheel_raw[0],
            "back_wheel": wheel_raw[1],
            "right_wheel": wheel_raw[2],
        }

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = abs(degps) * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        if degps < 0:
            return speed_int | 0x8000
        else:
            return speed_int & 0x7FFF
